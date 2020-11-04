from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class DoubleHeadRoIHead(StandardRoIHead):
    """RoI head for Double Head RCNN.

    https://arxiv.org/abs/1904.06493
    """

    def __init__(self, reg_roi_scale_factor, cls_roi_scale_factor=1.0, **kwargs):
        super(DoubleHeadRoIHead, self).__init__(**kwargs)
        self.reg_roi_scale_factor = reg_roi_scale_factor
        self.cls_roi_scale_factor = cls_roi_scale_factor

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing time."""
        """
        hr + double head:
        torch.Size([1, 256, 376, 672])
        torch.Size([1, 256, 188, 336])
        torch.Size([1, 256, 94, 168])
        torch.Size([1, 256, 47, 84])
        torch.Size([1, 256, 23, 42])

        efficient net:
        torch.Size([1, 64, 144, 256])
        torch.Size([1, 64, 72, 128])
        torch.Size([1, 64, 36, 64])
        torch.Size([1, 64, 18, 32])
        torch.Size([1, 64, 9, 16])
        """
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], 
            rois,
            roi_scale_factor=self.cls_roi_scale_factor)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results
