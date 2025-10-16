import os
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import laspy


# Hardcoded root directory to scan recursively for .laz files
ROOT_DIR: str = r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\custom-generated"

# Output directory for JSON results
OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "cluster_data")

# Classes to evaluate classification IoU on
TARGET_CLASSES = [0, 1, 2, 3, 4]

# Instance detection: which classes are considered as instances (exclude 0 terrain)
INSTANCE_CLASSES = [1, 2, 3, 4]

# IoU threshold for counting a matched instance as a true positive
IOU_THRESHOLD = 0.5


def read_dimension(points: laspy.ScaleAwarePointRecord, name: str) -> Optional[np.ndarray]:
    """
    Read a dimension (standard or extra) from LAS/LAZ points, if present.
    Returns None if not available.
    """
    # Standard dims
    if hasattr(points, name):
        try:
            arr = getattr(points, name)
            return np.asarray(arr)
        except Exception:
            pass

    # Extra dims
    try:
        if name in points.array.dtype.names:
            return np.asarray(points.array[name])
    except Exception:
        pass

    # laspy >=2 extra dims API
    try:
        ed = points.point_format.extra_dimension_names
        if ed and name in ed:
            return np.asarray(getattr(points, name))
    except Exception:
        pass

    return None


def safe_mode_label(labels: np.ndarray) -> int:
    if labels.size == 0:
        return -1
    counts = Counter(labels.tolist())
    return max(counts.items(), key=lambda kv: kv[1])[0]


def compute_confusion(true_labels: np.ndarray, pred_labels: np.ndarray, classes: List[int]) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def per_class_iou_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def build_instances(indices_by_id: Dict[int, np.ndarray], labels: np.ndarray) -> Dict[int, Tuple[Set[int], int]]:
    """
    Build instance dict: id -> (set_of_point_indices, majority_class_label)
    """
    instances: Dict[int, Tuple[Set[int], int]] = {}
    for inst_id, idxs in indices_by_id.items():
        inst_points = set(idxs.tolist())
        inst_class = safe_mode_label(labels[idxs])
        instances[inst_id] = (inst_points, inst_class)
    return instances


def index_by_instance_id(instance_ids: np.ndarray) -> Dict[int, np.ndarray]:
    mapping: Dict[int, np.ndarray] = {}
    if instance_ids is None:
        return mapping
    unique_ids = np.unique(instance_ids)
    for uid in unique_ids:
        # skip background-like ids if negative
        if uid < 0:
            continue
        mapping[int(uid)] = np.where(instance_ids == uid)[0]
    return mapping


def greedy_instance_matching(
    gt_instances: Dict[int, Tuple[Set[int], int]],
    pred_instances: Dict[int, Tuple[Set[int], int]],
    valid_classes: Iterable[int],
) -> List[Tuple[int, int, float, int]]:
    """
    Greedy match predicted to ground truth instances per class by IoU.
    Returns list of tuples: (gt_id, pred_id, iou, cls)
    """
    matches: List[Tuple[int, int, float, int]] = []
    valid_classes = set(valid_classes)

    # Group instances by class
    gt_by_class: Dict[int, List[int]] = defaultdict(list)
    pred_by_class: Dict[int, List[int]] = defaultdict(list)
    for gid, (gset, gcls) in gt_instances.items():
        if gcls in valid_classes:
            gt_by_class[gcls].append(gid)
    for pid, (pset, pcls) in pred_instances.items():
        if pcls in valid_classes:
            pred_by_class[pcls].append(pid)

    for cls in sorted(valid_classes):
        g_ids = gt_by_class.get(cls, [])
        p_ids = pred_by_class.get(cls, [])
        # Compute all pairwise IoUs
        pair_iou: List[Tuple[float, int, int]] = []  # (iou, gid, pid)
        for gid in g_ids:
            gset = gt_instances[gid][0]
            for pid in p_ids:
                pset = pred_instances[pid][0]
                inter = len(gset & pset)
                if inter == 0:
                    continue
                uni = len(gset | pset)
                iou = inter / uni if uni > 0 else 0.0
                pair_iou.append((iou, gid, pid))
        # Greedy selection by IoU descending
        pair_iou.sort(reverse=True, key=lambda x: x[0])
        used_gt: Set[int] = set()
        used_pred: Set[int] = set()
        for iou, gid, pid in pair_iou:
            if gid in used_gt or pid in used_pred:
                continue
            used_gt.add(gid)
            used_pred.add(pid)
            matches.append((gid, pid, iou, cls))
    return matches


def evaluate_file(path: str) -> Dict[str, object]:
    with laspy.open(path) as f:
        points = f.read()

    # Read labels
    gt_cls = read_dimension(points, "classification")
    pred_cls = read_dimension(points, "classification_prediction")

    gt_inst = read_dimension(points, "instance_id")
    pred_inst = read_dimension(points, "instance_id_prediction")

    if gt_cls is None or pred_cls is None:
        raise RuntimeError(f"Missing required classification fields in {path}")

    if gt_cls.shape[0] != pred_cls.shape[0]:
        raise RuntimeError("Ground truth and predicted classifications have different lengths")

    n = gt_cls.shape[0]
    # Ensure integer dtype
    gt_cls = gt_cls.astype(np.int64, copy=False)
    pred_cls = pred_cls.astype(np.int64, copy=False)

    # Classification IoU
    cm = compute_confusion(gt_cls, pred_cls, TARGET_CLASSES)
    per_class_iou = per_class_iou_from_cm(cm)
    miou = np.nanmean(per_class_iou)

    results: Dict[str, object] = {
        "num_points": int(n),
        "per_class_iou": {int(c): (None if np.isnan(per_class_iou[i]) else float(per_class_iou[i])) for i, c in enumerate(TARGET_CLASSES)},
        "classification_miou": float(miou) if not np.isnan(miou) else None,
    }

    # Instance metrics if fields available
    if gt_inst is not None and pred_inst is not None:
        gt_inst = gt_inst.astype(np.int64, copy=False)
        pred_inst = pred_inst.astype(np.int64, copy=False)

        gt_index = index_by_instance_id(gt_inst)
        pred_index = index_by_instance_id(pred_inst)

        gt_instances = build_instances(gt_index, gt_cls)
        pred_instances = build_instances(pred_index, pred_cls)

        # Filter out instances of classes not in INSTANCE_CLASSES or background-like
        gt_instances = {iid: v for iid, v in gt_instances.items() if v[1] in INSTANCE_CLASSES and len(v[0]) > 0}
        pred_instances = {iid: v for iid, v in pred_instances.items() if v[1] in INSTANCE_CLASSES and len(v[0]) > 0}

        matches = greedy_instance_matching(gt_instances, pred_instances, INSTANCE_CLASSES)
        matched_ious = [m[2] for m in matches]

        # mIoU over matched pairs (if none, None)
        inst_miou = float(np.mean(matched_ious)) if matched_ious else None

        # F1 at IOU_THRESHOLD
        tp = sum(1 for _, _, iou, _ in matches if iou >= IOU_THRESHOLD)
        fp = max(0, len(pred_instances) - tp)
        fn = max(0, len(gt_instances) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.update({
            "instance_detection_miou": inst_miou,
            "instance_detection_f1": f1,
            "instance_counts": {
                "gt": len(gt_instances),
                "pred": len(pred_instances),
                "matched": tp,
            },
        })
    else:
        results.update({
            "instance_detection_miou": None,
            "instance_detection_f1": None,
            "instance_counts": {"gt": 0, "pred": 0, "matched": 0},
        })

    return results


def discover_laz_files(root_dir: str) -> List[str]:
    laz_files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".laz"):
                laz_files.append(os.path.join(dirpath, fn))
    laz_files.sort()
    return laz_files


def merge_results(file_results: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Aggregate results across multiple files by summing confusion matrices approach.
    Since we only stored per-class IoU, recompute by concatenating would be best.
    To avoid re-reading files, during evaluation we will collect confusion as well.
    For simplicity here, we will compute weighted average by number of points for mIoU,
    and simple mean for instance metrics.
    """
    if not file_results:
        return {}

    # Weighted mIoU by points (approximation; exact requires aggregated CM)
    total_points = sum(fr.get("num_points", 0) for fr in file_results)
    if total_points == 0:
        weighted_miou = None
    else:
        weighted_miou = 0.0
        w_sum = 0
        for fr in file_results:
            nm = fr.get("classification_miou")
            npnts = fr.get("num_points", 0)
            if nm is not None and npnts > 0:
                weighted_miou += nm * npnts
                w_sum += npnts
        weighted_miou = (weighted_miou / w_sum) if w_sum > 0 else None

    # Average per-class IoU across files (ignoring None)
    per_class_vals: Dict[int, List[float]] = defaultdict(list)
    for fr in file_results:
        pci = fr.get("per_class_iou", {})
        for k, v in pci.items():
            if v is not None:
                per_class_vals[int(k)].append(float(v))
    per_class_mean = {k: (float(np.mean(v)) if v else None) for k, v in per_class_vals.items()}
    for c in TARGET_CLASSES:
        per_class_mean.setdefault(c, None)

    # Instance metrics: simple mean over files
    inst_mious = [fr.get("instance_detection_miou") for fr in file_results if fr.get("instance_detection_miou") is not None]
    inst_f1s = [fr.get("instance_detection_f1") for fr in file_results if fr.get("instance_detection_f1") is not None]
    avg_inst_miou = float(np.mean(inst_mious)) if inst_mious else None
    avg_inst_f1 = float(np.mean(inst_f1s)) if inst_f1s else None

    return {
        "files": len(file_results),
        "classification_miou_weighted": weighted_miou,
        "per_class_iou_mean": per_class_mean,
        "instance_detection_miou_mean": avg_inst_miou,
        "instance_detection_f1_mean": avg_inst_f1,
    }


def main():
    if not ROOT_DIR or not os.path.isdir(ROOT_DIR):
        print("No valid ROOT_DIR configured. Please set ROOT_DIR in this script.")
        return

    laz_files = discover_laz_files(ROOT_DIR)
    if not laz_files:
        print(f"No .laz files found under: {ROOT_DIR}")
        return

    # Collect all file results for aggregation
    all_file_results: List[Dict[str, object]] = []
    total_points = 0
    
    for p in laz_files:
        if not os.path.isfile(p):
            print(f"Skipping missing file: {p}")
            continue
        try:
            res = evaluate_file(p)
            all_file_results.append(res)
            total_points += res['num_points']
            print(f"Processed: {os.path.basename(p)} ({res['num_points']} points)")
        except Exception as e:
            print(f"Error evaluating {p}: {e}")

    if all_file_results:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Aggregate results
        agg = merge_results(all_file_results)
        print(f"\nAggregated results across {len(all_file_results)} files ({total_points} total points):")
        print(f"  Weighted classification mIoU: {agg['classification_miou_weighted']}")
        print("  Mean per-class IoU:")
        for cls in TARGET_CLASSES:
            v = agg["per_class_iou_mean"].get(cls)
            print(f"    Class {cls}: {v if v is not None else 'n/a'}")
        print(f"  Instance detection mIoU (mean): {agg['instance_detection_miou_mean']}")
        print(f"  Instance detection F1 (mean)@{IOU_THRESHOLD}: {agg['instance_detection_f1_mean']}")

        # Save JSON with only aggregated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"laz_metrics_{timestamp}.json")
        payload = {
            "root_dir": ROOT_DIR,
            "generated_at": timestamp,
            "target_classes": TARGET_CLASSES,
            "instance_classes": INSTANCE_CLASSES,
            "iou_threshold": IOU_THRESHOLD,
            "total_files": len(all_file_results),
            "total_points": total_points,
            "aggregate": agg,
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved metrics JSON: {out_path}")
        except Exception as e:
            print(f"Failed to save JSON metrics: {e}")


if __name__ == "__main__":
    main()


