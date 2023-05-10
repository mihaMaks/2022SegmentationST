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
                    field = field.replace(' - ', '|')
                    field = field.replace(' @@', '|')
                    field = field.replace(' ', '|')
                    field = field.lower()
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


def compute_stats(dists, overlaps, gold_lens, pred_lens, f1v2_tp, f1v2_fp, f1v2_fn, f1v3_tp, f1v3_fp, f1v3_fn, f1v4_tp, f1v4_fp, f1v4_fn):
    mean_dist = sum(dists) / len(dists)
    #f1_original (n_corect)
    total_overlaps = sum(overlaps)
    pred_lens_sum = sum(pred_lens)
    gold_lens_sum = sum(gold_lens)
    precision = 100 * total_overlaps / pred_lens_sum
    recall = 100 * total_overlaps / gold_lens_sum
    if precision+recall == 0:
        f_measure = .0
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    #f1_version2
    f1v2_tp_sum = sum(f1v2_tp)
    f1v2_pp_sum = sum(f1v2_fp) + f1v2_tp_sum
    f1v2_precision = 100 * f1v2_tp_sum / f1v2_pp_sum
    f1v2_gp_sum = sum(f1v2_fn) + f1v2_tp_sum
    f1v2_recall = 100 * f1v2_tp_sum / f1v2_gp_sum
    if f1v2_precision+f1v2_recall == 0:
        f1v2_score = .0
    else:
        f1v2_score = 2 * f1v2_precision * f1v2_recall / (f1v2_precision + f1v2_recall)

    #f1_version3
    f1v3_tp_sum = sum(f1v3_tp)
    f1v3_pp_sum = sum(f1v3_fp) + f1v3_tp_sum
    f1v3_precision = 100 * f1v3_tp_sum / f1v3_pp_sum
    f1v3_gp_sum = sum(f1v3_fn) + f1v3_tp_sum
    f1v3_recall = 100
    if(f1v3_gp_sum != 0):
        f1v3_recall = 100 * f1v3_tp_sum / f1v3_gp_sum

    if f1v3_precision+f1v3_recall == 0:
        f1v3_score = .0
    else:
        f1v3_score = 2 * f1v3_precision * f1v3_recall / (f1v3_precision + f1v3_recall)

    #f1_version4
    f1v4_tp_sum = sum(f1v4_tp)
    f1v4_pp_sum = sum(f1v4_fp) + f1v4_tp_sum
    f1v4_precision = 100 * f1v4_tp_sum / f1v4_pp_sum
    f1v4_gp_sum = sum(f1v4_fn) + f1v4_tp_sum
    f1v4_recall = 100
    if (f1v4_gp_sum != 0):
        f1v4_recall = 100 * f1v4_tp_sum / f1v4_gp_sum

    if f1v4_precision + f1v4_recall == 0:
        f1v4_score = .0
    else:
        f1v4_score = 2 * f1v4_precision * f1v4_recall / (f1v4_precision + f1v4_recall)

    return {"distance": mean_dist,  "f_ver1": f_measure, "f1_ver2": f1v2_score,"f1_ver3": f1v3_score, "f1_ver4": f1v4_score,
            "ver1_precision": precision, "ver2_precision": f1v2_precision, "ver3_precision": f1v3_precision, "f1v4_precision": f1v4_precision,
            "ver1_recall": recall, "ver2_recall": f1v2_recall, "ver3_recall": f1v3_recall, "ver4_recall": f1v4_recall}

"""
            "f1v1_pp_sum": pred_lens_sum, "f1v2_pp_sum": f1v2_pp_sum, "f1v3_pp_sum": f1v3_pp_sum, "f1v4_pp_sum": f1v4_pp_sum,
            "f1v1_gp_sum": gold_lens_sum, "f1v2_gp_sum": f1v2_gp_sum, "f1v3_gp_sum": f1v3_gp_sum, "f1v4_gp_sum": f1v4_gp_sum,
            "f1v1_tp_sum": total_overlaps, "f1v2_tp_sum": f1v2_tp_sum, "f1v3_tp_sum": f1v3_tp_sum, "f1v4_tp_sum": f1v4_tp_sum
"""
def stratify(sequence, labels):
    assert len(sequence) == len(labels)
    by_label = defaultdict(list)
    for label, value in zip(labels, sequence):
        by_label[label].append(value)
    return by_label
def f1_ver2(real_segm, pred_segm):
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

    true_positives = len(pred&real)
    false_positives = len(pred - real)
    false_negatives = len(real - pred)

    return true_positives, false_positives, false_negatives

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

    true_positives = len(pred&real)
    false_positives = len(pred - real)
    false_negatives = len(real - pred)

    if not real:
        # monomorph - 0 if something predicted, 1 otherwise
        true_positives = 0 if pred else 1
        false_negatives = 1 if pred else 0

    if not pred:
        false_positives = 1 if real else false_positives

    return true_positives, false_positives, false_negatives

def f1_ver4(real_segm, pred_segm):
    true_positives = false_positives = false_negatives = 0
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


def main(args):
    gold_data = read_tsv(args.gold, args.category)
    guess_data = read_tsv(args.guess, False)  # only second column is needed
    assert len(gold_data["segments"]) == len(guess_data["segments"]), \
        "gold and guess tsvs do not have the same number of entries"
    #f1_version2
    f1v2_tp = [f1_ver2(gold, guess)[0]
                       for gold, guess
                       in zip(gold_data["segments"], guess_data["segments"])]

    f1v2_fp = [f1_ver2(gold, guess)[1]
               for gold, guess
               in zip(gold_data["segments"], guess_data["segments"])]

    f1v2_fn = [f1_ver2(gold, guess)[2]
                       for gold, guess
                       in zip(gold_data["segments"], guess_data["segments"])]


    #f1_version3
    f1v3_tp = [f1_ver3(gold, guess)[0]
                       for gold, guess
                       in zip(gold_data["segments"], guess_data["segments"])]

    f1v3_fp = [f1_ver3(gold, guess)[1]
               for gold, guess
               in zip(gold_data["segments"], guess_data["segments"])]

    f1v3_fn = [f1_ver3(gold, guess)[2]
                       for gold, guess
                       in zip(gold_data["segments"], guess_data["segments"])]
    #f1_ver4
    f1v4_tp = [f1_ver4(gold, guess)[0]
               for gold, guess
               in zip(gold_data["segments"], guess_data["segments"])]

    f1v4_fp = [f1_ver4(gold, guess)[1]
               for gold, guess
               in zip(gold_data["segments"], guess_data["segments"])]

    f1v4_fn = [f1_ver4(gold, guess)[2]
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

        f1v2_overlaps_by_cat = stratify(f1v2_tp, categories)
        f1v2_gold_ovelaps_by_cat = stratify(f1v2_fp, categories)
        f1v2_pred_overlaps_by_cat = stratify(f1v2_fn, categories)

        f1v3_tp_by_cat = stratify(f1v3_tp, categories)
        f1v3_fp_by_cat = stratify(f1v3_fp, categories)
        f1v3_fn_by_cat = stratify(f1v3_fn, categories)

        f1v4_tp_by_cat = stratify(f1v4_tp, categories)
        f1v4_fp_by_cat = stratify(f1v4_fp, categories)
        f1v4_fn_by_cat = stratify(f1v4_fn, categories)

        for cat in sorted(dists_by_cat):
            cat_stats = compute_stats(
                dists_by_cat[cat],
                overlaps_by_cat[cat],
                gold_lens_by_cat[cat],
                pred_lens_by_cat[cat],

                f1v2_overlaps_by_cat[cat],
                f1v2_gold_ovelaps_by_cat[cat],
                f1v2_pred_overlaps_by_cat[cat],

                f1v3_tp_by_cat[cat],
                f1v3_fp_by_cat[cat],
                f1v3_fn_by_cat[cat],

                f1v4_tp_by_cat[cat],
                f1v4_fp_by_cat[cat],
                f1v4_fn_by_cat[cat]


            )
            print_numbers(cat_stats, cat=cat)

    overall_stats = compute_stats(dists, n_overlaps, gold_lens, pred_lens, f1v2_tp, f1v2_fp, f1v2_fn,
                                                f1v3_tp, f1v3_fp, f1v3_fn, f1v4_tp, f1v4_fp, f1v4_fn)
    print_numbers(overall_stats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SIGMORPHON 2022 Morpheme Segmentation Shared Task Evaluation')
    parser.add_argument("--gold", help="Gold standard", required=True, type=str)
    parser.add_argument("--guess", help="Model output", required=True, type=str)
    parser.add_argument("--category", help="Morphological category", action="store_true")
    opt = parser.parse_args()
    main(opt)
