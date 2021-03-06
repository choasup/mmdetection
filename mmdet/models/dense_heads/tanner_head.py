import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, multihead_nms
from ..builder import HEADS, build_loss, build_head
from .anchor_free_head import AnchorFreeHead
from .base_dense_head import BaseDenseHead

from mmdet.core.bbox.assigners.detr_assigner import build_matcher
from mmdet.models.losses import accuracy
from mmdet.core.utils.dist_utils import (is_dist_avail_and_initialized,
                                         get_world_size, accuracy, interpolate)
from mmdet.core.utils import box_ops

INF = 1e8

@HEADS.register_module()
class TannerHead(nn.Module):
    def __init__(self, num_heads, num_classes, **kwargs):
        super(TannerHead, self).__init__()        
        self.num_classes = num_classes
        self.test_cfg = kwargs['test_cfg']
        self.heads = nn.ModuleList()
        for idx in range(num_heads):
            sub_head = kwargs["sub_bbox_head_{}".format(idx + 1)]
            if kwargs['train_cfg'] != None:
                sub_head.update(train_cfg=kwargs['train_cfg']["train_cfg_sub_{}".format(idx + 1)])
            sub_head.update(test_cfg=kwargs['test_cfg']) 
            self.heads.append(build_head(sub_head))
        self.matcher = build_matcher()
        
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
        weight_dict['loss_giou'] = 2
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(self.num_classes, matcher=self.matcher, weight_dict=weight_dict,
                                 eos_coef=0.1, losses=losses)
 
    def init_weights(self):
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        losses = {}
        rois = []
       
        for idx, sub_head in enumerate(self.heads): 
            outs = sub_head.forward_train(x, 
                                          img_metas, 
                                          gt_bboxes, 
                                          gt_labels, 
                                          gt_bboxes_ignore, 
                                          proposal_cfg,
                                          **kwargs)
            if proposal_cfg:
                loss, out = outs
                rois.append(out)
            else:
                loss = outs

            name = type(sub_head).__name__
            for key, value in loss.items():
                losses["sub{}_{}_{}".format(idx, name, key)] = value

        # for matcher 
        img_res = [[[], []] for i in range(len(img_metas))]
        for sub_head_box in rois:
            for idx, res in enumerate(sub_head_box):
                box, label = res
                img_res[idx][0].append(box)
                img_res[idx][1].append(label) 
            
        cat_bboxes = []
        cat_labels = []    
        for idx, res in enumerate(img_res):
            cat_bboxes.append(torch.cat(res[0]).unsqueeze(dim=0))
            cat_labels.append(torch.cat(res[1]).unsqueeze(dim=0))

        # torch.Size([4, 22400, 4])
        out_bboxes = torch.cat(cat_bboxes)
        # torch.Size([4, 22400, 81])
        out_labels = torch.cat(cat_labels)

        for img_info, mlvl_bbox in zip(img_metas, out_bboxes):
            mlvl_bbox[:, 0] = mlvl_bbox[:, 0] / img_info['pad_shape'][1]
            mlvl_bbox[:, 1] = mlvl_bbox[:, 1] / img_info['pad_shape'][0]
            mlvl_bbox[:, 2] = mlvl_bbox[:, 2] / img_info['pad_shape'][1]
            mlvl_bbox[:, 3] = mlvl_bbox[:, 3] / img_info['pad_shape'][0]

        outputs = {'pred_logits': out_labels, 'pred_boxes': out_bboxes}
        targets = []
        for img_info, gt_box, gt_label in zip(img_metas, gt_bboxes, gt_labels):
            gt_box[:, 0] = gt_box[:, 0] / img_info['pad_shape'][1]
            gt_box[:, 1] = gt_box[:, 1] / img_info['pad_shape'][0]
            gt_box[:, 2] = gt_box[:, 2] / img_info['pad_shape'][1]
            gt_box[:, 3] = gt_box[:, 3] / img_info['pad_shape'][0]
            
            target = {}
            target['boxes'] = gt_box
            target['labels'] = gt_label
            targets.append(target)
 
        loss_matcher = self.criterion(outputs, targets)
        losses.update(loss_matcher)

        return losses       

    def forward(self, feats):
        outs = []
        for idx, sub_head in enumerate(self.heads):
            out = sub_head.forward(feats)
            outs.append(out)
        return [outs]

    def get_bboxes(self, outs, img_metas, cfg=None, rescale=None):
        cfg = self.test_cfg if cfg is None else cfg
        bboxes = []
        labels = []
        tanners = []
        for idx, (sub_head, out) in enumerate(zip(self.heads, outs)): 
            #if idx != 4:
            #    continue
            sub_bbox_list = sub_head.get_bboxes(*out, img_metas, rescale=rescale)       
            for bbox, label in sub_bbox_list:
                bboxes.append(bbox)
                labels.append(label)
                tanners.append(torch.zeros_like(label) + idx)
        
        cat_bboxes = torch.cat(bboxes)
        cat_labels = torch.cat(labels) 
        cat_tanners = torch.cat(tanners).reshape(-1, 1).to(dtype=torch.float)

        # nms output
        if True:
            # multi-head nms
            mlvl_bboxes = cat_bboxes[:, :4]
            value_scores = cat_bboxes[:, -1:].reshape(-1)
            mlvl_scores = torch.zeros([cat_bboxes.shape[0], self.num_classes + 1]).to(device=mlvl_bboxes.device)
            
            index_x = torch.where(cat_labels > -1)[0]
            index_y = cat_labels
            index = (index_x, index_y)
            mlvl_scores[index] = value_scores
           
            if True:
                cfg.nms['iou_threshold'] = 0.5
                det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
            if False:
                det_bboxes, det_labels = multihead_nms(mlvl_bboxes, mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)      
  
            bbox_list = [(det_bboxes, det_labels)]
        
        # no-nms output
        if False:
            bbox_list = [(cat_bboxes, cat_labels)]

        return bbox_list
         
    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
