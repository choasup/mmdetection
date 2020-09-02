import torch
from mmcv.ops.nms import batched_nms


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]

"""
Implementation of Merge NMS
"""
def torch_nms(bboxes, threshold = 0.5):
    """Pure Torch NMS.(isn't used in this code)"""  
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    scores = bboxes[:,4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending = True)
    keep = []

    while order.numel():
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1 + 1).clamp(min=0) * (yy2 - yy1 + 1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter )
        idx = (iou <= threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    return torch.tensor(keep)

def nms(dets, scores, thresh, thresh2 = 0.9):  
    """Numpy NMS implemented by myself."""  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    # scores = dets[:, 4]  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]  
    keep = []  
    keep_scores = []
    while order.size > 0:   
        i = order[0]  
        keep.append(int(i))  
        # keep_scores.append( scores[i] )
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  

        # cal new score
        new_score_inds = np.where(ovr > thresh2)[0]
        new_score_order = order[new_score_inds + 1] 
        score_list = np.append( scores[new_score_order], scores[i] )
        new_score = scores[i] + (score_list**4).sum()
        # scores[i] + (score_list**4).sum()   # (1 - np.prod(1 - score_list)) ** (1/len(score_list))
        keep_scores.append( new_score  )
        # origin score
        #keep_scores.append( scores[i] )

        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  

    return np.array(keep), np.array(keep_scores)


def box_results_with_nms_and_limit(scores, boxes, threshold = 0.6, SCORE_THRESH = 0.05, DETECTIONS_PER_IM = 100): 
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).
    """
    num_classes = 80
    cls_boxes = [[] for _ in range(num_classes)]
    for j in range( num_classes ):   
        inds = np.where(scores[:, j] > SCORE_THRESH)[0] # 0.05
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, :]
        # dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        if boxes_j.any():
            # dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
            # keep = box_utils.nms(dets_j, cfg.TEST.NMS) # 0.3
            # keep = nms( torch.tensor(dets_j), threshold = threshold)
            keep, keep_scores = nms( boxes_j, scores_j, threshold)
            nms_dets = np.hstack((boxes_j[keep, :], keep_scores[:, np.newaxis])).astype(np.float32, copy=False)
        else:
            nms_dets = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        cls_boxes[j] = nms_dets

    if DETECTIONS_PER_IM > 0: 
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(num_classes)]  
        )
        if len(image_scores) > DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-DETECTIONS_PER_IM]
            for j in range( num_classes): 
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack( [cls_boxes[j] for j in range(num_classes)] )
    
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes

def multihead_nms(multi_bboxes,
                  multi_scores,
                  score_thr,
                  nms_cfg,
                  max_num=-1,
                  score_factors=None):

    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    # (TODO). betas nms
    #dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]
