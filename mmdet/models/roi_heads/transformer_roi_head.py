import torch
from torch import nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

from .roi_extractors.transformer_roi_extractor import build_transformer
from .roi_extractors.position_encoding import build_position_encoding


@HEADS.register_module()
class TransformerRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(TransformerRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg) 

        self.transformer = build_transformer() 
        self.position_encoding = build_position_encoding()
        
        # hidden_dim: 256
        self.input_proj = nn.Conv2d(bbox_head.in_channels, 256, kernel_size=1)
        self.query_embed = nn.Embedding(100, 256)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass
 
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        #nn.init.normal_(self.input_proj.weight, 0, 0.01)
        #nn.init.normal_(self.query_embed.weight, 0, 0.01)

        if self.with_bbox:
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        raise NotImplementedError

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x,
                                                    proposal_list,
                                                    gt_bboxes,
                                                    gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, proposal_lists,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        # pos_feats: torch.Size([2, 256, 76, 50])
        
        #bbox_feats = self.bbox_roi_extractor(
        #    x[:self.bbox_roi_extractor.num_inputs], rois)
        #if self.with_shared_head:
        #    bbox_feats = self.shared_head(bbox_feats)

        src = x[0]
        pos, mask = self.position_encoding(src)
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0] 
      
        bbox_results = self.bbox_head(hs) 
        return bbox_results

    def _bbox_forward_train(self, x, proposal_list, 
                            gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        # TODO. not the real rois 
        rois = proposal_list
        bbox_results = self._bbox_forward(x, rois)

        # TODO. box need to normalize
        for idx, (label, box) in enumerate(zip(gt_labels, gt_bboxes)):
            img_metas[idx]['labels'] = label
            img_metas[idx]['boxes'] = box

        loss_bbox = self.bbox_head.loss(bbox_results,
                                        rois,
                                        img_metas)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        raise NotImplementedError

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        #assert self.with_bbox, 'Bbox head must be implemented.'

        #det_bboxes, det_labels = self.simple_test_bboxes(
        #    x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        #bbox_results = bbox2result(det_bboxes, det_labels,
        #                           self.bbox_head.num_classes)

        #if not self.with_mask:
        #    return bbox_results
        #else:
        #    segm_results = self.simple_test_mask(
        #        x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #    return bbox_results, segm_results
        raise NotImplementedError

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        ## recompute feats to save memory
        #det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
        #                                              proposal_list,
        #                                              self.test_cfg)

        #if rescale:
        #    _det_bboxes = det_bboxes
        #else:
        #    _det_bboxes = det_bboxes.clone()
        #    _det_bboxes[:, :4] *= det_bboxes.new_tensor(
        #        img_metas[0][0]['scale_factor'])
        #bbox_results = bbox2result(_det_bboxes, det_labels,
        #                           self.bbox_head.num_classes)

        ## det_bboxes always keep the original scale
        #if self.with_mask:
        #    segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
        #                                      det_labels)
        #    return bbox_results, segm_results
        #else:
        #    return bbox_results
        raise NotImplementedError
