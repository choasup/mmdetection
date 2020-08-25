from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class TANNERS(SingleStageDetector):
    """Implementation of `TANNERS <https://arxiv.org/abs/?>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TANNERS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
