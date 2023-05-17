import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from unidecode import unidecode

# write a .csv file :
# command: python3 word_by_word_eval.py --guess ../baseline/eng.word.dev.bert.tsv --gold ../data/eng.word.dev.tsv --out word_by_word_metrics.csv
"""
columns:        index|word|segmentation|prediction|distance|n_correct|f1_segmentation

run metrics for every word and write them in a .csv
"""


def compare_len(real_segm, pred_segm):
    real = set()
    pred = set()
    c = 0
    for s in real_segm.split('|'):
        real.add((c, s))
        c += len(s)
    r_len = c

    c = 0
    for s in pred_segm.split('|'):
        pred.add((c, s))
        c += len(s)
    p_len = c

    if r_len != p_len:
        return False
    return True


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
    if not compare_len(gold_segments, guess_segments):
        return -1

    a = gold_segments.split("|")
    b = guess_segments.split("|")
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def f1_ver2(real_segm, pred_segm):
    if not compare_len(real_segm, pred_segm):
        return 0, 0, 0

    real_segm = real_segm.lower()
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

    true_positives = len(pred & real)
    false_positives = len(pred - real)
    false_negatives = len(real - pred)

    return true_positives, false_positives, false_negatives


def f1_ver3(real_segm, pred_segm):
    true_positives = false_positives = false_negatives = 0
    if not compare_len(real_segm, pred_segm):
        return 0, 0, 0

    letter_pairs_r = set()
    letter_pairs_p = set()
    r = 1
    for i, l in enumerate(real_segm):
        if l == '|':
            letter_pairs_r.add(i - r)
            r += 1
    p = 1
    for i, l in enumerate(pred_segm):
        if l == '|':
            letter_pairs_p.add(i - p)
            p += 1
    true_positives = len(letter_pairs_p & letter_pairs_r)
    false_positives = len(letter_pairs_p - letter_pairs_r)
    false_negatives = len(letter_pairs_r - letter_pairs_p)

    # word is monomorph
    if not letter_pairs_r:
        true_positives = 0 if letter_pairs_p else 1
        false_negatives = 1 if letter_pairs_p else 0
    # word is predicted as monomprh
    if not letter_pairs_p:
        false_positives = 1 if letter_pairs_r else false_positives

    return true_positives, false_positives, false_negatives


# ROW MANIPULATION
def appendData(ground, pred_seg):
    row_dat = list()
    row_dat.append(ground[2])  # category
    row_dat.append(ground[0])  # word
    # segmentation
    grnd = ground[1]
    grnd = grnd.replace(" - ", "|")
    grnd = grnd.replace(" @@", "|")
    grnd = grnd.replace(" ", "|")
    grnd = unidecode(grnd)
    row_dat.append(grnd.lower())

    # prediction
    pred = pred_seg
    pred = pred.replace(" - ", "|")
    pred = pred.replace(" @@", "|")
    pred = pred.replace(" ", "|")
    pred = unidecode(pred)
    row_dat.append(pred)
    return row_dat


def appendMetrics(ground, pred_seg):
    row_met = list()
    gold_segments = ground[1].replace(" - ", "|")
    gold_segments = gold_segments.replace(" @@", "|")
    gold_segments = gold_segments.replace(" ", "|")
    gold_segments = unidecode(gold_segments)

    guess_segments = pred_seg.replace(" - ", "|")
    guess_segments = guess_segments.replace(" @@", "|")
    guess_segments = guess_segments.replace(" ", "|")
    guess_segments = unidecode(guess_segments)
    gold_segments = gold_segments.lower()
    row_met.append(distance(gold_segments, guess_segments))

    row_met.append(n_correct(gold_segments, guess_segments))

    # callculate instance-f1-score
    tp_fp_fn_v2 = f1_ver2(gold_segments, guess_segments)
    precision = 0
    if tp_fp_fn_v2[1] + tp_fp_fn_v2[0] != 0:
        precision = tp_fp_fn_v2[0] / (tp_fp_fn_v2[0] + tp_fp_fn_v2[1])
    recall = 0
    if tp_fp_fn_v2[2] + tp_fp_fn_v2[0] != 0:
        recall = tp_fp_fn_v2[0] / (tp_fp_fn_v2[0] + tp_fp_fn_v2[2])
    if precision + recall == 0:
        f1_score_v2 = .0
    else:
        f1_score_v2 = 2 * precision * recall / (precision + recall)
    row_met.append(tp_fp_fn_v2[0])
    row_met.append(tp_fp_fn_v2[1])
    row_met.append(tp_fp_fn_v2[2])

    # callculate instance-f1-score
    tp_fp_fn_v3 = f1_ver3(gold_segments, guess_segments)
    precision = 0
    if tp_fp_fn_v3[1] + tp_fp_fn_v3[0] != 0:
        precision = tp_fp_fn_v3[0] / (tp_fp_fn_v3[0] + tp_fp_fn_v3[1])
    recall = 0
    if tp_fp_fn_v3[2] + tp_fp_fn_v3[0] != 0:
        recall = tp_fp_fn_v3[0] / (tp_fp_fn_v3[0] + tp_fp_fn_v3[2])
    if precision + recall == 0:
        f1_score_v3 = .0
    else:
        f1_score_v3 = 2 * precision * recall / (precision + recall)
    row_met.append(tp_fp_fn_v3[0])
    row_met.append(tp_fp_fn_v3[1])
    row_met.append(tp_fp_fn_v3[2])

    row_met.append(round(f1_score_v2, 3))
    row_met.append(round(f1_score_v3, 3))
    row_met.append(abs(round((f1_score_v3 - f1_score_v2), 3)))

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
                                                                 'distance', 'overlaps',
                                                                 'f1v2_tp', 'f1v2_fp', 'f1v2_fn',
                                                                 'f1v3_tp', 'f1v3_fp', 'f1v3_fn',
                                                                 'f1v2_score', 'f1v3_score', 'abs_diff'])

    table_data_frame['category'] = table_data_frame['category'].apply('_{:03}'.format)
    table_data_frame.to_csv(args.out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Word by word metrics')
    parser.add_argument("--gold", help="Gold standard", required=True, type=str)
    parser.add_argument("--guess", help="Model output", required=True, type=str)
    parser.add_argument("--out", help="Output file name", required=True, type=str)
    opt = parser.parse_args()
    main(opt)
