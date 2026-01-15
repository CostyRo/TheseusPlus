import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.normalized_entropy_cpd import NormalizedEntropyCPD
from src.utils.metrics import metricor


NECPD_NAME_BASE = "NECPD"
NECPD_MEASURES = (
    "AUC_ROC",
    "AUC_PR",
    "Precision",
    "Recall",
    "F",
    "Precision@k",
    "Rprecision",
    "Rrecall",
    "RF",
)


def _resolve_paths(dataset_folder, filename):
    folder = dataset_folder
    if "NASA_" in folder:
        folder = folder.replace("NASA_", "NASA-")
        ts_name = filename.replace("SMAP", "").replace("_data.out", ".test.out")
    else:
        ts_name = filename.replace(".txt", ".out")

    ts_path = os.path.join("data", "benchmark_new", folder, ts_name)
    return folder, ts_name, ts_path + ".zip"


def _normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    denom = scores.max() - scores.min()
    if denom == 0:
        return scores
    return (scores - scores.min()) / denom


def main():
    parser = argparse.ArgumentParser(
        description="Precompute NECPD scores + metric values for all benchmark time series."
    )
    parser.add_argument(
        "--table",
        default=os.path.join("data", "mergedTable_AUC_PR.csv"),
        help="CSV used to list benchmark files (default: data/mergedTable_AUC_PR.csv).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "new_algs"),
        help="Output root directory (default: data/new_algs).",
    )
    parser.add_argument(
        "--windows",
        default="70,100",
        help="Comma-separated window sizes to precompute (default: 70,100).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing score files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N time series (0 = no limit).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.table)
    if "dataset" not in df.columns or "filename" not in df.columns:
        raise ValueError(f"Missing 'dataset'/'filename' columns in {args.table}")

    windows = []
    for part in str(args.windows).split(","):
        part = part.strip()
        if not part:
            continue
        windows.append(int(part))
    windows = sorted(set(windows))
    if not windows:
        raise ValueError("--windows must contain at least one integer")

    items = df[["dataset", "filename"]].drop_duplicates().to_dict("records")
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    methods = [(f"{NECPD_NAME_BASE}{w}", w) for w in windows]
    metrics_rows_by_method = {name: [] for name, _ in methods}

    grader = metricor()
    total = len(items)
    for idx, item in enumerate(items, 1):
        dataset = str(item["dataset"])
        filename = str(item["filename"])
        folder, ts_name, ts_zip = _resolve_paths(dataset, filename)

        print(f"[{idx}/{total}] {dataset} / {filename}")

        if not os.path.isfile(ts_zip):
            print(f"  - missing: {ts_zip}")
            for method_name, window in methods:
                row = {"dataset": dataset, "filename": filename, "window": int(window)}
                for m in NECPD_MEASURES:
                    row[m] = np.nan
                metrics_rows_by_method[method_name].append(row)
            continue

        try:
            ts = pd.read_csv(ts_zip, compression="zip", header=None).to_numpy()
        except Exception as exc:
            print(f"  - failed to read: {ts_zip} ({exc})")
            for method_name, window in methods:
                row = {"dataset": dataset, "filename": filename, "window": int(window)}
                for m in NECPD_MEASURES:
                    row[m] = np.nan
                metrics_rows_by_method[method_name].append(row)
            continue

        if ts.size == 0:
            for method_name, window in methods:
                row = {"dataset": dataset, "filename": filename, "window": int(window)}
                for m in NECPD_MEASURES:
                    row[m] = np.nan
                metrics_rows_by_method[method_name].append(row)
            continue

        label = np.asarray(ts[:, 1], dtype=int)
        data = ts[:, 0].astype(float)

        for method_name, window_raw in methods:
            out_dir = os.path.join(args.out, method_name)
            scores_root = os.path.join(out_dir, "scores")
            score_out_dir = os.path.join(scores_root, folder, "score")
            score_out_zip = os.path.join(score_out_dir, f"{ts_name}.zip")

            row = {"dataset": dataset, "filename": filename, "window": int(window_raw)}
            for m in NECPD_MEASURES:
                row[m] = np.nan

            window = max(2, min(int(window_raw), len(data)))
            row["window"] = window

            try:
                if args.overwrite or not os.path.isfile(score_out_zip):
                    os.makedirs(score_out_dir, exist_ok=True)
                    clf = NormalizedEntropyCPD(window=window, bins="ln", score="delta")
                    clf.fit(data)
                    score = _normalize_scores(clf.decision_scores_)
                    pd.DataFrame(score).to_csv(
                        score_out_zip, index=False, header=False, compression="zip"
                    )
                score = (
                    pd.read_csv(score_out_zip, compression="zip", header=None)
                    .to_numpy()[:, 0]
                    .astype(float)
                )
            except Exception as exc:
                print(f"  - failed score compute/save for {method_name}: {exc}")
                metrics_rows_by_method[method_name].append(row)
                continue

            if np.sum(label) in (0, len(label)):
                metrics_rows_by_method[method_name].append(row)
                continue

            try:
                _, _, ap = grader.metric_PR(label, score)
                row["AUC_PR"] = float(ap)
            except Exception:
                pass

            try:
                L = grader.metric_new(label, score)
                if L is not None:
                    row["AUC_ROC"] = float(L[0])
                    row["Precision"] = float(L[1])
                    row["Recall"] = float(L[2])
                    row["F"] = float(L[3])
                    row["Rrecall"] = float(L[4])
                    row["Rprecision"] = float(L[7])
                    row["RF"] = float(L[8])
                    row["Precision@k"] = float(L[9])
            except Exception:
                pass

            metrics_rows_by_method[method_name].append(row)

    for method_name, _ in methods:
        out_dir = os.path.join(args.out, method_name)
        os.makedirs(out_dir, exist_ok=True)
        out_metrics = os.path.join(out_dir, "metrics.csv")
        out_df = pd.DataFrame(metrics_rows_by_method[method_name])
        out_df.to_csv(out_metrics, index=False)
        print(f"Saved: {out_metrics}")


if __name__ == "__main__":
    main()
