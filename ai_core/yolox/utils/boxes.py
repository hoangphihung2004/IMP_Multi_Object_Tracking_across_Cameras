#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np
import torch
import torchvision
from numba import njit

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]

def filter_box(output, scale_range):
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]

@njit
def bboxes_iou(a, b): # Đã đổi tên từ matrix_iou thành bboxes_iou
    n = a.shape[0]
    m = b.shape[0]
    ious = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        area_a = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
        for j in range(m):
            area_b = (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1])
            xx1 = max(a[i, 0], b[j, 0])
            yy1 = max(a[i, 1], b[j, 1])
            xx2 = min(a[i, 2], b[j, 2])
            yy2 = min(a[i, 3], b[j, 3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            union = area_a + area_b - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious

@njit
def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    res = bbox.copy()
    for i in range(res.shape[0]):
        res[i, 0] = res[i, 0] * scale_ratio + padw
        res[i, 2] = res[i, 2] * scale_ratio + padw
        res[i, 1] = res[i, 1] * scale_ratio + padh
        res[i, 3] = res[i, 3] * scale_ratio + padh
    return res

@njit
def xyxy2xywh(bboxes):
    res = bboxes.copy()
    for i in range(res.shape[0]):
        res[i, 2] = res[i, 2] - res[i, 0]
        res[i, 3] = res[i, 3] - res[i, 1]
    return res

@njit
def xyxy2cxcywh(bboxes):
    res = bboxes.copy()
    for i in range(res.shape[0]):
        w = res[i, 2] - res[i, 0]
        h = res[i, 3] - res[i, 1]
        res[i, 0] = res[i, 0] + w * 0.5
        res[i, 1] = res[i, 1] + h * 0.5
        res[i, 2] = w
        res[i, 3] = h
    return res

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        
        if len(detections.shape) > 1 and detections.shape[1] == 1:
            detections = detections.squeeze(0)
            
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        output[i] = detections if output[i] is None else torch.cat((output[i], detections))

    return output
