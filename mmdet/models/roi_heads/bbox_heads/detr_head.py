import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.core.bbox.assigners.detr_assigner import build_matcher

from mmdet.core.utils.dist_utils import (is_dist_avail_and_initialized,
                                         get_world_size, accuracy, interpolate)
from mmdet.core.utils import box_ops


@HEADS.register_module()
class DetrHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 hidden_dim=256,
                 aux_loss=True):
        super(DetrHead, self).__init__()
        assert with_cls or with_reg
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.num_classes = num_classes
        self.aux_loss = aux_loss

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.init_loss()

    def init_loss(self):
        self.matcher = build_matcher()

        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
        weight_dict['loss_giou'] = 2
        if False:
            weight_dict["loss_mask"] = 1
        weight_dict["loss_dice"] = 1
        # TODO this is a hack
        if True:
            aux_weight_dict = {}
            for i in range(6 - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']
        if False:
            losses += ["masks"]
        self.criterion = SetCriterion(self.num_classes, matcher=self.matcher, weight_dict=weight_dict,
                                 eos_coef=0.1, losses=losses)

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.class_embed.weight, 0, 0.01)
            nn.init.constant_(self.class_embed.bias, 0)
        #if self.with_reg:
        #    nn.init.normal_(self.bbox_embed.weight, 0, 0.001)
        #    nn.init.constant_(self.bbox_embed.bias, 0)

    @auto_fp16()
    def forward(self, hs):
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
 
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        #num_pos = pos_bboxes.size(0)
        #num_neg = neg_bboxes.size(0)
        #num_samples = num_pos + num_neg

        ## original implementation uses new_zeros since BG are set to be 0
        ## now use empty & fill because BG cat_id = num_classes,
        ## FG cat_id = [0, num_classes-1]
        #labels = pos_bboxes.new_full((num_samples, ),
        #                             self.num_classes,
        #                             dtype=torch.long)
        #label_weights = pos_bboxes.new_zeros(num_samples)
        #bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        #bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        #if num_pos > 0:
        #    labels[:num_pos] = pos_gt_labels
        #    pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        #    label_weights[:num_pos] = pos_weight
        #    if not self.reg_decoded_bbox:
        #        pos_bbox_targets = self.bbox_coder.encode(
        #            pos_bboxes, pos_gt_bboxes)
        #    else:
        #        pos_bbox_targets = pos_gt_bboxes
        #    bbox_targets[:num_pos, :] = pos_bbox_targets
        #    bbox_weights[:num_pos, :] = 1
        #if num_neg > 0:
        #    label_weights[-num_neg:] = 1.0

        #return labels, label_weights, bbox_targets, bbox_weights
        raise NotImplementedError

    def get_targets(self,
                    gt_bboxes,
                    gt_labels,
                    img_metas,
                    concat=False):
        for idx, (label, box) in enumerate(zip(gt_labels, gt_bboxes)):
            w = img_metas[idx]['pad_shape'][1]
            h = img_metas[idx]['pad_shape'][0]
            box = box / torch.tensor([w, h, w, h], device=box.device)
            img_metas[idx]['labels'] = label
            img_metas[idx]['boxes'] = box
        return img_metas

    def loss(self,
             outputs,
             rois,
             targets):
        
        loss_dict = self.criterion(outputs, targets) 
        return loss_dict

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        #if isinstance(cls_score, list):
        #    cls_score = sum(cls_score) / float(len(cls_score))
        #scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        #if bbox_pred is not None:
        #    bboxes = self.bbox_coder.decode(
        #        rois[:, 1:], bbox_pred, max_shape=img_shape)
        #else:
        #    bboxes = rois[:, 1:].clone()
        #    if img_shape is not None:
        #        bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
        #        bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        #if rescale and bboxes.size(0) > 0:
        #    if isinstance(scale_factor, float):
        #        bboxes /= scale_factor
        #    else:
        #        scale_factor = bboxes.new_tensor(scale_factor)
        #        bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
        #                  scale_factor).view(bboxes.size()[0], -1)

        #if cfg is None:
        #    return bboxes, scores
        #else:
        #    det_bboxes, det_labels = multiclass_nms(bboxes, scores,
        #                                            cfg.score_thr, cfg.nms,
        #                                            cfg.max_per_img)

        #    return det_bboxes, det_labels
        raise NotImplementedError

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        #img_ids = rois[:, 0].long().unique(sorted=True)
        #assert img_ids.numel() <= len(img_metas)

        #bboxes_list = []
        #for i in range(len(img_metas)):
        #    inds = torch.nonzero(
        #        rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
        #    num_rois = inds.numel()

        #    bboxes_ = rois[inds, 1:]
        #    label_ = labels[inds]
        #    bbox_pred_ = bbox_preds[inds]
        #    img_meta_ = img_metas[i]
        #    pos_is_gts_ = pos_is_gts[i]

        #    bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
        #                                   img_meta_)

        #    # filter gt bboxes
        #    pos_keep = 1 - pos_is_gts_
        #    keep_inds = pos_is_gts_.new_ones(num_rois)
        #    keep_inds[:len(pos_is_gts_)] = pos_keep

        #    bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        #return bboxes_list
        raise NotImplementedError

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        #assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        #if not self.reg_class_agnostic:
        #    label = label * 4
        #    inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
        #    bbox_pred = torch.gather(bbox_pred, 1, inds)
        #assert bbox_pred.size(1) == 4

        #if rois.size(1) == 4:
        #    new_rois = self.bbox_coder.decode(
        #        rois, bbox_pred, max_shape=img_meta['img_shape'])
        #else:
        #    bboxes = self.bbox_coder.decode(
        #        rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
        #    new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        #return new_rois
        raise NotImplementedError

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

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

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
