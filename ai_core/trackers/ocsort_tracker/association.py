import os
import numpy as np
from numba import njit

@njit
def iou_batch(bboxes1, bboxes2):
    n = bboxes1.shape[0]
    m = bboxes2.shape[0]
    ious = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(m):
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            xx1 = max(bboxes1[i, 0], bboxes2[j, 0])
            yy1 = max(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = min(bboxes1[i, 2], bboxes2[j, 2])
            yy2 = min(bboxes1[i, 3], bboxes2[j, 3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            wh = w * h
            union = area1 + area2 - wh
            if union > 0:
                ious[i, j] = wh / union
    return ious

@njit
def giou_batch(bboxes1, bboxes2):
    n = bboxes1.shape[0]
    m = bboxes2.shape[0]
    gious = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(m):
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            xx1 = max(bboxes1[i, 0], bboxes2[j, 0])
            yy1 = max(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = min(bboxes1[i, 2], bboxes2[j, 2])
            yy2 = min(bboxes1[i, 3], bboxes2[j, 3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            wh = w * h
            union = area1 + area2 - wh
            iou = wh / union if union > 0 else 0.0
            
            xxc1 = min(bboxes1[i, 0], bboxes2[j, 0])
            yyc1 = min(bboxes1[i, 1], bboxes2[j, 1])
            xxc2 = max(bboxes1[i, 2], bboxes2[j, 2])
            yyc2 = max(bboxes1[i, 3], bboxes2[j, 3])
            area_enclose = (xxc2 - xxc1) * (yyc2 - yyc1)
            giou = iou - (area_enclose - union) / area_enclose if area_enclose > 0 else iou
            gious[i, j] = (giou + 1.0) / 2.0
    return gious

@njit
def diou_batch(bboxes1, bboxes2):
    n = bboxes1.shape[0]
    m = bboxes2.shape[0]
    dious = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
        for j in range(m):
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            xx1 = max(bboxes1[i, 0], bboxes2[j, 0])
            yy1 = max(bboxes1[i, 1], bboxes2[j, 1])
            xx2 = min(bboxes1[i, 2], bboxes2[j, 2])
            yy2 = min(bboxes1[i, 3], bboxes2[j, 3])
            wh = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            union = area1 + area2 - wh
            iou = wh / union if union > 0 else 0.0
            
            cx1, cy1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0, (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
            cx2, cy2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0, (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            inner_diag = (cx1 - cx2)**2 + (cy1 - cy2)**2
            
            xxc1, yyc1 = min(bboxes1[i, 0], bboxes2[j, 0]), min(bboxes1[i, 1], bboxes2[j, 1])
            xxc2, yyc2 = max(bboxes1[i, 2], bboxes2[j, 2]), max(bboxes1[i, 3], bboxes2[j, 3])
            outer_diag = (xxc2 - xxc1)**2 + (yyc2 - yyc1)**2
            diou = iou - inner_diag / outer_diag if outer_diag > 0 else iou
            dious[i, j] = (diou + 1.0) / 2.0
    return dious

@njit
def ciou_batch(bboxes1, bboxes2):
    n = bboxes1.shape[0]
    m = bboxes2.shape[0]
    cious = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        w1, h1 = bboxes1[i, 2] - bboxes1[i, 0], bboxes1[i, 3] - bboxes1[i, 1]
        area1 = w1 * h1
        for j in range(m):
            w2, h2 = bboxes2[j, 2] - bboxes2[j, 0], bboxes2[j, 3] - bboxes2[j, 1]
            area2 = w2 * h2
            xx1, yy1 = max(bboxes1[i, 0], bboxes2[j, 0]), max(bboxes1[i, 1], bboxes2[j, 1])
            xx2, yy2 = min(bboxes1[i, 2], bboxes2[j, 2]), min(bboxes1[i, 3], bboxes2[j, 3])
            wh = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            union = area1 + area2 - wh
            iou = wh / union if union > 0 else 0.0
            
            cx1, cy1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0, (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
            cx2, cy2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0, (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            dist = (cx1 - cx2)**2 + (cy1 - cy2)**2
            xxc1, yyc1 = min(bboxes1[i, 0], bboxes2[j, 0]), min(bboxes1[i, 1], bboxes2[j, 1])
            xxc2, yyc2 = max(bboxes1[i, 2], bboxes2[j, 2]), max(bboxes1[i, 3], bboxes2[j, 3])
            diag = (xxc2 - xxc1)**2 + (yyc2 - yyc1)**2
            
            arctan = np.arctan(w2 / (h2 + 1e-6)) - np.arctan(w1 / (h1 + 1e-6))
            v = (4.0 / (np.pi**2)) * (arctan**2)
            alpha = v / (1.0 - iou + v + 1e-6)
            ciou = iou - dist / (diag + 1e-6) - alpha * v
            cious[i, j] = (ciou + 1.0) / 2.0
    return cious

@njit
def ct_dist(bboxes1, bboxes2):
    n, m = bboxes1.shape[0], bboxes2.shape[0]
    dists = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        cx1, cy1 = (bboxes1[i, 0] + bboxes1[i, 2]) / 2.0, (bboxes1[i, 1] + bboxes1[i, 3]) / 2.0
        for j in range(m):
            cx2, cy2 = (bboxes2[j, 0] + bboxes2[j, 2]) / 2.0, (bboxes2[j, 1] + bboxes2[j, 3]) / 2.0
            dists[i, j] = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    if dists.size > 0:
        max_d = np.max(dists)
        if max_d > 0: dists = dists / max_d
        return 1.0 - dists
    return dists

@njit
def speed_direction_batch(dets, tracks):
    n, m = dets.shape[0], tracks.shape[0]
    dy, dx = np.zeros((m, n), dtype=np.float32), np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        cx2, cy2 = (tracks[i, 0] + tracks[i, 2]) / 2.0, (tracks[i, 1] + tracks[i, 3]) / 2.0
        for j in range(n):
            cx1, cy1 = (dets[j, 0] + dets[j, 2]) / 2.0, (dets[j, 1] + dets[j, 3]) / 2.0
            diff_x, diff_y = cx1 - cx2, cy1 - cy2
            norm = np.sqrt(diff_x**2 + diff_y**2) + 1e-6
            dx[i, j], dy[i, j] = diff_x / norm, diff_y / norm
    return dy, dx

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[int(m[0]), int(m[1])]<iou_threshold):
            unmatched_detections.append(int(m[0]))
            unmatched_trackers.append(int(m[1]))
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):    
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y_exp = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X_exp = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = np.clip(inertia_X_exp * X + inertia_Y_exp * Y, -1.0, 1.0)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)] = 0
    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask_exp = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    angle_diff_cost = (valid_mask_exp * diff_angle).T * vdc_weight * scores
    if min(iou_matrix.shape) > 0:
        matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0,2))
    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:,1]]
    matches = []
    for m in matched_indices:
        if iou_matrix[int(m[0]), int(m[1])] < iou_threshold:
            unmatched_detections.append(int(m[0]))
            unmatched_trackers.append(int(m[1]))
        else:
            matches.append(m.reshape(1,2))
    if len(matches) == 0: matches = np.empty((0,2), dtype=int)
    else: matches = np.concatenate(matches, axis=0).astype(np.int32)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def associate_kitti(detections, trackers, det_cates, iou_threshold, 
        velocities, previous_obs, vdc_weight):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    """
        Cost from the velocity direction consistency
    """
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y_exp = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X_exp = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = np.clip(inertia_X_exp * X + inertia_Y_exp * Y, -1.0, 1.0)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)]=0  
    valid_mask_exp = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = (valid_mask_exp * diff_angle).T * vdc_weight * scores

    """
        Cost from IoU
    """
    iou_matrix = iou_batch(detections, trackers)
    
    """
        With multiple categories, generate the cost for catgory mismatch
    """
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
            for j in range(num_trk):
                if det_cates[i] != trackers[j, 4]:
                        cate_matrix[i][j] = -1e6
    
    cost_matrix = - iou_matrix - angle_diff_cost - cate_matrix

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[int(m[0]), int(m[1])]<iou_threshold):
            unmatched_detections.append(int(m[0]))
            unmatched_trackers.append(int(m[1]))
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)