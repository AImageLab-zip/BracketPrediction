#!/usr/bin/env python3
"""Score prediction file.
    - mAP
    - mAR
"""

import argparse
import json
import pandas as pd
import pickle
from metrics import eval_map, voc_ar, calculate_metrics_per_scan
import numpy as np
import debugpy


NEW_LANDMARKS = ["Bracket", "Planar", "Incisal"]


def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file", type=str, required=True)
    parser.add_argument("-g", "--goldstandard_file", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="results.json")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    return parser.parse_args()


def score(gt_all, pred_all_map):
    """
    Calculate metrics for: AP at different distance threshold
    """
    score_dict = {}
    dist_thresh_list = []
    recall = {
        "Mesial": [],
        "Distal": [],
        "Cusp": [],
        "InnerPoint": [],
        "OuterPoint": [],
        "FacialPoint": []
    }
    for i in range(0, 30):
        dist_thresh = 0.1 * i
        rec, prec, ap = eval_map(pred_all_map, gt_all, dist_thresh=dist_thresh)
        score_dict[str(i)] = ap
        dist_thresh_list.append(dist_thresh)
        for cat in rec.keys():
            recall[cat].append(rec[cat][-1])
        # Collect all values for each class
    class_values = {'Mesial': [], 'Distal': [], 'Cusp': [], 'InnerPoint': [], 'OuterPoint': [], 'FacialPoint': []}
    for threshold in score_dict.values():
        for class_name in class_values.keys():
            class_values[class_name].append(threshold[class_name])

    # Calculate mean AP per class across thresholds
    map = {class_name: sum(values) / len(values) for class_name, values in class_values.items()}

    # Calculate AR per class
    mar = {}
    for cat in recall.keys():
        ar = voc_ar(np.exp(-np.asarray(dist_thresh_list)), recall, cat)
        mar[cat] = ar

    all_metrics = {"AP": map, "AR": mar}
    return all_metrics


def reformat_scores(scores):
    fmt_scores = {}
    metrics_cat = scores['AP']
    fmt_scores["AP_cusp"] = metrics_cat['Cusp']
    fmt_scores["AP_mesial_distal"] = (metrics_cat['Mesial'] + metrics_cat['Distal'])/2
    fmt_scores["AP_inner_outer"] = (metrics_cat['InnerPoint'] + metrics_cat['OuterPoint'])/2
    fmt_scores["AP_facial"] = metrics_cat['FacialPoint']
    fmt_scores["mAP"] = (metrics_cat['Cusp'] + metrics_cat['Mesial'] + metrics_cat['Distal'] +
                         metrics_cat['InnerPoint'] + metrics_cat['OuterPoint']+ metrics_cat['FacialPoint']) / 6

    metrics_cat = scores['AR']
    fmt_scores["AR_cusp"] = metrics_cat['Cusp']
    fmt_scores["AR_mesial_distal"] = (metrics_cat['Mesial'] + metrics_cat['Distal'])/2
    fmt_scores["AR_inner_outer"] = (metrics_cat['InnerPoint'] + metrics_cat['OuterPoint'])/2
    fmt_scores["AR_facial"] = metrics_cat['FacialPoint']
    fmt_scores["mAR"] = (metrics_cat['Cusp'] + metrics_cat['Mesial'] + metrics_cat['Distal'] +
                         metrics_cat['InnerPoint'] + metrics_cat['OuterPoint'] + metrics_cat['FacialPoint']) / 6

    return fmt_scores


def main():
    """Main function."""
    args = get_args()
    if args.debug:
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Waiting for debugger on port 5681 …")
        debugpy.wait_for_client()
        print(">>> Debugger attached.")
    pred_submission = pd.read_csv(
        args.predictions_file
    )

    pred_all_map = {
        "Mesial": {},
        "Distal": {},
        "Cusp": {},
        "InnerPoint": {},
        "OuterPoint": {},
        "FacialPoint": {}
    }

    for _, row in pred_submission.iterrows():
        class_name = row['class']
        key = row['key']
        coord = [row['coord_x'], row['coord_y'], row['coord_z']]
        prob = row['score']
        if class_name in NEW_LANDMARKS: continue
        if key not in pred_all_map[class_name]:
            pred_all_map[class_name][key] = [[coord, prob]]
        else:
            pred_all_map[class_name][key].append([coord, prob])

    with open(args.goldstandard_file, 'rb') as fp:
        gold = pickle.load(fp)
    scores = score(gold, pred_all_map)
    scores = reformat_scores(scores)

    # Compute per-scan metrics to list worst-performing samples
    # filter out NEW_LANDMARKS so per-scan calc doesn't KeyError
    filtered_pred = pred_submission[~pred_submission['class'].isin(NEW_LANDMARKS)].reset_index(drop=True)
    per_scan_metrics = calculate_metrics_per_scan(filtered_pred, gold)

    # Helper to extract reformatted per-scan metric values
    def compute_scan_metric(scan_metrics, metric_name):
        # scan_metrics: {'mAP': {class: val...}, 'mAR': {...}}
        if metric_name.startswith('AP_'):
            if metric_name == 'AP_cusp':
                return scan_metrics['mAP']['Cusp']
            if metric_name == 'AP_mesial_distal':
                return (scan_metrics['mAP']['Mesial'] + scan_metrics['mAP']['Distal']) / 2
            if metric_name == 'AP_inner_outer':
                return (scan_metrics['mAP']['InnerPoint'] + scan_metrics['mAP']['OuterPoint']) / 2
            if metric_name == 'AP_facial':
                return scan_metrics['mAP']['FacialPoint']
            if metric_name == 'AP_mAP':
                vals = list(scan_metrics['mAP'].values())
                return sum(vals) / len(vals)
        if metric_name.startswith('AR_'):
            if metric_name == 'AR_cusp':
                return scan_metrics['mAR']['Cusp']
            if metric_name == 'AR_mesial_distal':
                return (scan_metrics['mAR']['Mesial'] + scan_metrics['mAR']['Distal']) / 2
            if metric_name == 'AR_inner_outer':
                return (scan_metrics['mAR']['InnerPoint'] + scan_metrics['mAR']['OuterPoint']) / 2
            if metric_name == 'AR_facial':
                return scan_metrics['mAR']['FacialPoint']
            if metric_name == 'AR_mAR':
                vals = list(scan_metrics['mAR'].values())
                return sum(vals) / len(vals)
        return None

    metric_keys = [
        'AP_cusp', 'AP_mesial_distal', 'AP_inner_outer', 'AP_facial', 'AP_mAP',
        'AR_cusp', 'AR_mesial_distal', 'AR_inner_outer', 'AR_facial', 'AR_mAR'
    ]

    std_scores = {}
    for mk in metric_keys:
        vals = []
        for _, scan_metrics in per_scan_metrics.items():
            val = compute_scan_metric(scan_metrics, mk)
            if val is not None:
                vals.append(float(val))
        if vals:
            out_name = mk.replace('AP_', 'std_AP_').replace('AR_', 'std_AR_')
            std_scores[out_name] = float(np.std(vals))

    worst_samples = {}
    for mk in metric_keys:
        scores_list = []
        for scan, scan_metrics in per_scan_metrics.items():
            val = compute_scan_metric(scan_metrics, mk)
            if val is None:
                continue
            scores_list.append((scan, float(val)))
        # sort ascending -> worst first
        scores_list.sort(key=lambda x: x[1])
        worst5 = [{'scan': s, 'score': v} for s, v in scores_list[:5]]
        worst_samples[mk] = worst5

    # Print worst samples summary
    print('\nWorst 5 samples per metric:')
    for mk, items in worst_samples.items():
        print(f"\n{mk}:")
        for it in items:
            print(f"  {it['scan']}: {it['score']:.6f}")

    # Write results including worst samples
    with open(args.output, "w") as out:
        res = {"submission_status": "SCORED", **scores, **std_scores, "worst_samples": worst_samples}
        out.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()


# Example GT structure that must be saved to pickle
""" gold = {
    "Mesial": {
        "patient_001": [[1.0, 2.0, 3.0]],
        "patient_002": [[1.1, 2.1, 3.1]]
    },
    "Distal": {
        "patient_001": [[1.5, 2.5, 3.5]],
        "patient_002": [[1.6, 2.6, 3.6]]
    },
    "Cusp": {
        "patient_001": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "patient_002": [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
    },
    "InnerPoint": {
        "patient_001": [[2.0, 2.0, 2.0]],
        "patient_002": [[2.1, 2.1, 2.1]]
    },
    "OuterPoint": {
        "patient_001": [[3.0, 3.0, 3.0]],
        "patient_002": [[3.1, 3.1, 3.1]]
    },
    "FacialPoint": {
        "patient_001": [[4.0, 4.0, 4.0]],
        "patient_002": [[4.1, 4.1, 4.1]]
    }
} """

""" with open("gold.pkl", "wb") as f:
    pickle.dump(gold, f) """
