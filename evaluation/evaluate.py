#!/usr/bin/env python
"""
Official Evaluation Script for the SIGMORPHON 2022 Morpheme Segmentation Shared Task.
Returns precision, recall, f-measure, and mean Levenhstein distance.
"""

from collections import defaultdict
import numpy as np


def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2) + 1, len(str1) + 1], dtype=int)
    for x in range(1, len(str2) + 1):
        m[x, 0] = m[x - 1, 0] + 1
    for y in range(1, len(str1) + 1):
        m[0, y] = m[0, y - 1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x, y] = min(m[x - 1, y] + 1, m[x, y - 1] + 1, m[x - 1, y - 1] + dg)
    return m[len(str2), len(str1)]


def print_numbers(stats_dict, cat="all"):
    print("\n")
    print("category:", cat)
                        #sorted(stats_dict.items())
    for stat_name, stat in stats_dict.items():
        print("\t".join([stat_name, "{:.3f}".format(stat)]))


def read_tsv(path, category):
    # tsv without header
    col_names = ["word", "segments"]
    if category:
        col_names.append("category")
    data = {name: [] for name in col_names}
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            for name, field in zip(col_names, fields):
                if name == "segments":
                    field = field.replace(' @@', '|')
                    field = field.replace(' ', '|')
                data[name].append(field)
    return data


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


def compute_stats(dists, overlaps, gold_lens, pred_lens, f1v2_sc_pp_gp_tp, f1v3_sc_pp_gp_tp):
    mean_dist = sum(dists) / len(dists)
    #n_corect
    total_overlaps = sum(overlaps)
    pred_lens_sum = sum(pred_lens)
    gold_lens_sum = sum(gold_lens)
    precision = 100 * total_overlaps / pred_lens_sum
    recall = 100 * total_overlaps / gold_lens_sum
    if precision+recall == 0:
        f_measure = .0
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    #f1_segments
    f1v2_score = sum([score[0] for score in f1v2_sc_pp_gp_tp]) / len(f1v2_sc_pp_gp_tp)
    f1v2_tp_sum = sum([score[3] for score in f1v2_sc_pp_gp_tp])
    f1v2_precision = 100 * f1v2_tp_sum /pred_lens_sum
    f1v2_recall = 100 * f1v2_tp_sum /gold_lens_sum

    #f1_blocks
    f1v3_score = sum([score[0] for score in f1v3_sc_pp_gp_tp]) / len(f1v3_sc_pp_gp_tp)
    f1v3_tp_sum = sum([score[3] for score in f1v3_sc_pp_gp_tp])
    f1v3_precision = 100 * f1v3_tp_sum / sum([score[1] for score in f1v3_sc_pp_gp_tp])
    f1v3_recall = 100 * f1v3_tp_sum / sum([score[2] for score in f1v3_sc_pp_gp_tp])

    return {"distance": mean_dist,  "f_ver1": f_measure, "f1_ver2": f1v2_score,"f1_ver3": f1v3_score,
            "ver1_precision": precision, "ver2_precision": f1v2_precision, "ver3_precision": f1v3_precision,
            "ver1_recall": recall, "ver2_recall": f1v2_recall, "ver3_recall": f1v3_recall,
            "f1v1_sum": total_overlaps, "f1v2_sum": f1v2_tp_sum, "f1v3_sum": f1v3_tp_sum}


def stratify(sequence, labels):
    assert len(sequence) == len(labels)
    by_label = defaultdict(list)
    for label, value in zip(labels, sequence):
        by_label[label].append(value)
    return by_label
def f1_ver2(real_segm, pred_segm):
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
    recall = 100 * len(pred & real) / len(real)
    precision = 100 * len(pred & real) / len(pred)
    pred_positives = len(pred)
    grnd_positives = len(real)
    true_positives = len(pred&real)

    if precision + recall == 0:
        return (0, pred_positives, grnd_positives, true_positives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return (f1_score, pred_positives, grnd_positives, true_positives)
def f1_ver3(true, predicted):
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
        return (0, pred_positives, grnd_positives, true_positives)
    f2_score = 2 * (precision * recall) / (precision + recall)
    return (f2_score, pred_positives, grnd_positives, true_positives)



def main(args):
    gold_data = read_tsv(args.gold, args.category)
    guess_data = read_tsv(args.guess, False)  # only second column is needed
    assert len(gold_data["segments"]) == len(guess_data["segments"]), \
        "gold and guess tsvs do not have the same number of entries"
    #new metrics
    f1v2_sc_pp_gp_tp = [f1_ver2(gold, guess)
                       for gold, guess
                       in zip(gold_data["segments"], guess_data["segments"])]
    f1v3_sc_pp_gp_tp = [f1_ver3(gold, guess)
                       for gold, guess
                       in zip(gold_data["segments"], guess_data["segments"])]
    # levenshtein distance can be computed separately for each pair
    dists = [distance(gold, guess)
             for gold, guess
             in zip(gold_data["segments"], guess_data["segments"])]

    # the values needed for P/R can also be broken down per-example
    n_overlaps = [n_correct(gold, guess)
                  for gold, guess
                  in zip(gold_data["segments"], guess_data["segments"])]
    gold_lens = [len(gold.split("|")) for gold in gold_data["segments"]]
    pred_lens = [len(guess.split("|")) for guess in guess_data["segments"]]

    if args.category:
        categories = gold_data["category"]
        # stratify by category
        dists_by_cat = stratify(dists, categories)
        overlaps_by_cat = stratify(n_overlaps, categories)
        gold_lens_by_cat = stratify(gold_lens, categories)
        pred_lens_by_cat = stratify(pred_lens, categories)
        pred_f1v2_by_cat = stratify(f1v2_sc_pp_gp_tp, categories)
        pred_f1v3_by_cat = stratify(f1v3_sc_pp_gp_tp, categories)

        for cat in sorted(dists_by_cat):
            cat_stats = compute_stats(
                dists_by_cat[cat],
                overlaps_by_cat[cat],
                gold_lens_by_cat[cat],
                pred_lens_by_cat[cat],
                pred_f1v2_by_cat[cat],
                pred_f1v3_by_cat[cat]
            )
            print_numbers(cat_stats, cat=cat)

    overall_stats = compute_stats(dists, n_overlaps, gold_lens, pred_lens, f1v2_sc_pp_gp_tp, f1v2_sc_pp_gp_tp)
    print_numbers(overall_stats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SIGMORPHON 2022 Morpheme Segmentation Shared Task Evaluation')
    parser.add_argument("--gold", help="Gold standard", required=True, type=str)
    parser.add_argument("--guess", help="Model output", required=True, type=str)
    parser.add_argument("--category", help="Morphological category", action="store_true")
    opt = parser.parse_args()
    main(opt)
