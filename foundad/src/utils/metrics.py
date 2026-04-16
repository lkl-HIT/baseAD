import numpy as np
import torch
from adeval import EvalAccumulatorCuda


def compute_ad_metrics_gpu(
    image_scores: np.ndarray,
    image_labels: np.ndarray,
    pixel_scores: np.ndarray,
    pixel_masks: np.ndarray,
    nstrips: int = 200,
) -> dict:
    """Compute all AD metrics on GPU via adeval's EvalAccumulatorCuda.

    Parameters
    ----------
    image_scores : (N,) float  – per-image anomaly scores (normalised).
    image_labels : (N,) int    – per-image ground truth (0=normal, 1=anomaly).
    pixel_scores : (N, H, W) float – pixel anomaly maps (normalised).
    pixel_masks  : (N, H, W) int   – pixel ground truth masks.
    nstrips      : histogram bins for the accumulator.

    Returns
    -------
    dict with keys: inst_auroc, inst_aupr, pix_auroc, pix_aupr, pro_auc
    """
    score_min, score_max = float(image_scores.min()), float(image_scores.max())
    anomap_min, anomap_max = float(pixel_scores.min()), float(pixel_scores.max())

    accum = EvalAccumulatorCuda(
        score_min, score_max,
        anomap_min, anomap_max,
        skip_pixel_aupro=False,
        nstrips=nstrips,
    )

    for i in range(len(image_scores)):
        accum.add_image(
            torch.tensor(float(image_scores[i])),
            torch.tensor(int(image_labels[i])),
        )

    accum.add_anomap_batch(
        torch.from_numpy(pixel_scores.astype(np.float32)).cuda(non_blocking=True),
        torch.from_numpy(pixel_masks.astype(np.uint8)).cuda(non_blocking=True),
    )

    m = accum.summary()

    return {
        "inst_auroc": m["i_auroc"],
        "inst_aupr": m["i_aupr"],
        "pix_auroc": m["p_auroc"],
        "pix_aupr": m["p_aupr"],
        "pro_auc": m["p_aupro"],
    }
