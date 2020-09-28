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
def std_nms(bboxes, scores, threshold = 0.5):
    """Pure Torch NMS.(isn't used in this code)"""  
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
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

def high_nms(bboxes, scores, threshold = 0.9):
    """Pure Torch NMS.(isn't used in this code)"""
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending = True)
    keep = []

    while order.numel():
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break

        i = order[0].item()
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1 + 1).clamp(min=0) * (yy2 - yy1 + 1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter )
        idx = (iou <= 0.5).nonzero().squeeze()

        # update scores
        # part norm = 3, AP 36.0
        # part norm = 2, AP 36.2
        # all norm = 2, AP 36.2
        idh = (iou > threshold).nonzero().squeeze()
        if idh.numel() > 0:
            scores[i] = (scores[i] ** 2 + torch.norm(scores[order[idh + 1]], p=2) ** 2) ** (1.0 / 2)
            keep.append(i)
 
        if idx.numel() == 0:
            break
        order = order[idx + 1]

    if keep == []:
        keep = [0]
    return torch.tensor(keep)


def cross_nms(bboxes, scores, threshold = 0.6):
    """Pure Torch NMS.(isn't used in this code)"""
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending = True)
    keep = []
 
    while order.numel():
        if order.numel() == 1:
            break

        i = order[0].item()
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1 + 1).clamp(min=0) * (yy2 - yy1 + 1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter )
        idx = (iou <= threshold).nonzero().squeeze()
        
        # Give high scores
        idh = (iou > 0.9).nonzero().squeeze()
        if idh.numel() > 2:
            keep.append(i)

        if idx.numel() == 0:
            break
        order = order[idx + 1]
    
    return torch.tensor(keep)

def merge_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    #nms_type = nms_cfg_.pop('type', 'nms')
    #nms_op = eval(nms_type)
    #dets, keep = high_nms(boxes_for_nms, scores, **nms_cfg_)
   
    keep = high_nms(boxes_for_nms, scores)
    boxes = boxes[keep]
    scores = scores[keep]

    """
    keep = cross_nms(boxes_for_nms, scores)
    if len(keep) != 0:
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        keep = torch.tensor([0])
        boxes = boxes[keep]
        scores = torch.tensor([0.0]).to(device=scores.device)
    """
 
    return torch.cat([boxes, scores[:, None]], -1), keep

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
    dets, keep = merge_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]
