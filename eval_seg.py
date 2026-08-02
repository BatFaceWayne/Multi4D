"""Faithful port of TRASE metrics_segmentation.py core: per-frame binary IoU + pixel
accuracy of a predicted object mask vs GT, averaged -> mIoU / mAcc.

Usage: python eval_seg.py <pred_masks_dir> <gt_masks_dir>
Matches masks by filename (GT drives the frame set). Pred masks may be RGB (non-black
= foreground) or grayscale; GT is a boolean/0-255 PNG.
"""
import os, sys
import numpy as np
from PIL import Image


def compute_acc(pred, gt):
    return np.sum(pred == gt) / gt.size


def compute_iou(pred, gt):
    intersection = np.sum(np.logical_and(pred, gt))
    union = np.sum(np.logical_or(pred, gt))
    return 0.0 if union == 0 else intersection / union


def to_bool(arr):
    if arr.ndim == 3:
        return (arr[..., :3].mean(axis=-1) / 255.0).astype(bool)   # RGB render -> non-black
    return arr > 127 if arr.dtype != bool else arr


def main(pred_dir, gt_dir):
    ious, accs, n = [], [], 0
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        pf = os.path.join(pred_dir, fname)
        if not os.path.exists(pf):
            pf = os.path.join(pred_dir, os.path.splitext(fname)[0] + ".png")
            if not os.path.exists(pf):
                continue
        gt = to_bool(np.asarray(Image.open(os.path.join(gt_dir, fname))))
        pred = to_bool(np.asarray(Image.open(pf)))
        if pred.shape != gt.shape:  # guard resolution mismatch
            pred = np.asarray(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]))).astype(bool)
        accs.append(compute_acc(pred, gt))
        ious.append(compute_iou(pred, gt))
        n += 1
    if n == 0:
        print("no matching frames between pred and gt"); return
    print(f"frames={n}  mIoU={np.mean(ious):.4f}  mAcc={np.mean(accs):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python eval_seg.py <pred_mask_dir> <gt_mask_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
