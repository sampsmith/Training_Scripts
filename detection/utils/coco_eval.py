from typing import Any, Dict, List, Tuple

import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def convert_to_coco_results(
    predictions: List[Dict[str, Any]],
    image_ids: List[int],
    contiguous_to_cat_id: Dict[int, int],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for img_id, pred in zip(image_ids, predictions):
        if len(pred["boxes"]) == 0:
            continue
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        # Convert xyxy -> xywh
        xywh = copy.deepcopy(boxes)
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
        for box, score, label in zip(xywh, scores, labels):
            result = {
                "image_id": int(img_id),
                "category_id": int(contiguous_to_cat_id[int(label)]),
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ],
                "score": float(score),
            }
            results.append(result)
    return results


def coco_evaluate(
    coco_gt: COCO,
    predictions: List[Dict[str, Any]],
    image_ids: List[int],
    contiguous_to_cat_id: Dict[int, int],
) -> Tuple[Dict[str, float], COCOeval]:
    if len(predictions) == 0:
        return {"mAP": 0.0}, None

    coco_results = convert_to_coco_results(predictions, image_ids, contiguous_to_cat_id)
    if len(coco_results) == 0:
        return {"mAP": 0.0}, None

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # type: ignore[attr-defined]
    metrics = {
        "AP": float(stats[0]),  # mAP @[.5:.95]
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }
    return metrics, coco_eval


