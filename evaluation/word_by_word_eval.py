import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# write a .csv file :
# command: python3 word_by_word_eval.py --guess ../baseline/eng.word.dev.bert.tsv --gold ../data/eng.word.dev.tsv --out word_by_word_metrics.csv
"""
columns:        index|word|segmentation|prediction|distance|n_correct|f1_segmentation

run metrics for every word and write them in a .csv
"""


# METRICS
def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2) + 1, len(str1) + 1], dtype=int)
    for x in range(1, len(str2) + 1):
        m[x, 0] = m[x - 1, 0] + 1
    for y in range(1, len(str1) + 1):
        m[0, y] = m[0, y - 1] + 1

    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y - 1] == str2[x - 1]:
                dg = 0
            else:
                dg = 1
            m[x, y] = min(m[x - 1, y] + 1, m[x, y - 1] + 1, m[x - 1, y - 1] + dg)
    return m[len(str2), len(str1)]

def n_correct(gold_segments, guess_segments):
    a = gold_segments.split("|")
    b = guess_segments.split("|")
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]

def f1_segmentation(real_segm, pred_segm):
    real = set()
    pred = set()
    c = 0
    for s in real_segm.split('|'):
        real.add((c, s))
        c += len(s)
    c = 0
    for s in pred_segm.split('|'):
        pred.add((c, s))
        c += len(s)
    recall = len(pred & real) / len(real)
    precision = len(pred & real) / len(pred)
    if precision + recall == 0:
        return (len(real&pred), 0)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return (len(real&pred), f1_score)

def f1_breaks(true, predicted):
    real = set()
    pred = set()
    c = 0
    for s in true.split("|")[:-1]:
        c += len(s)
        real.add(c)
    c = 0
    for s in predicted.split("|")[:-1]:
        c += len(s)
        pred.add(c)
    grnd_positives = len(real)
    pred_positives = len(pred)
    true_positives = len(pred & real)
    # if prediction is correct count as if it is counting segments
    if pred_positives == grnd_positives and true_positives == grnd_positives:
        true_positives += 1
        grnd_positives += 1
        pred_positives += 1
    else:  # or just count as if last segment is guessed incorectly
        grnd_positives += 1
        pred_positives += 1
    if real:
        recall = 100 * len(pred & real) / len(real)
    else:
        # monomorph - 0 if something predicted, 1 otherwise
        recall = 0 if pred else 100
        grnd_positives = 1
        pred_positives = len(pred) if pred else 1

    if pred:
        precision = 100 * len(pred & real) / len(pred)
    else:
        # predicted monomorph - 0 if it should not be, 1 if it is
        precision = 0 if real else 100
        true_positives = 0 if real else 1
        pred_positives = 1


    if precision + recall == 0:
        return (true_positives, 0)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return (true_positives, f1_score)

# ROW MANIPULATION
def appendData(ground, pred_seg):
    row_dat = list()
    row_dat.append(ground[2])  # category
    row_dat.append(ground[0])  # word
    row_dat.append(ground[1].lower())  # segmentation
    row_dat.append(pred_seg)  # prediction
    return row_dat


def appendMetrics(ground, pred_seg):
    row_met = list()
    gold_segments = ground[1].replace(" @@", "|")
    gold_segments = gold_segments.replace(" ", "|")
    guess_segments = pred_seg.replace(" @@", "|")
    guess_segments = guess_segments.replace(" ", "|")
    gold_segments = gold_segments.lower()
    row_met.append(distance(gold_segments, guess_segments))
    row_met.append(n_correct(gold_segments, guess_segments))
    for met in f1_segmentation(gold_segments, guess_segments):
        row_met.append(met)
    for met in f1_breaks(gold_segments, guess_segments):
        row_met.append(met)

    return row_met

def main(args):
    ground_truth = pd.read_table(args.gold, names=['word', 'segmentation', 'category'], dtype={'category': str})
    predicted_segmentation = pd.read_table(args.guess, usecols=[1], names=['segmentation'])
    assert len(ground_truth["segmentation"]) == len(predicted_segmentation["segmentation"]), \
        "gold and guess tsvs do not have the same number of entries"

    word_by_word_table = list()
    for ground, pred_seg in zip(ground_truth.itertuples(False, None), predicted_segmentation['segmentation']):
        row_data = appendData(ground, pred_seg)
        row_metrics = appendMetrics(ground, pred_seg)
        row_data.extend(row_metrics)
        word_by_word_table.append(row_data)



    word_by_word_table.sort()
    table_data_frame = pd.DataFrame(word_by_word_table, columns=['category', 'word', 'segmentation', 'prediction',
                                                                 'distance','overlaps', 'f1v2_overlaps', 'f1v2_score',
                                                                 'f1v3_overlaps', 'f1v3_score'])

    table_data_frame['category'] = table_data_frame['category'].apply('_{:03}'.format)
    table_data_frame['f1v2_score'] = table_data_frame['f1v2_score'].apply('{:0.2f}'.format)
    table_data_frame['f1v3_score'] = table_data_frame['f1v3_score'].apply('{:0.2f}'.format)
    table_data_frame.to_csv(args.out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Word by word metrics')
    parser.add_argument("--gold", help="Gold standard", required=True, type=str)
    parser.add_argument("--guess", help="Model output", required=True, type=str)
    parser.add_argument("--out", help="Output file name", required=True, type=str)
    opt = parser.parse_args()
    main(opt)
