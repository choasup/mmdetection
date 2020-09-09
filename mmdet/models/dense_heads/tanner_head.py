import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, multihead_nms
from ..builder import HEADS, build_loss, build_head
from .anchor_free_head import AnchorFreeHead
from .base_dense_head import BaseDenseHead

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
        for idx, sub_head in enumerate(self.heads): 
            loss = sub_head.forward_train(x, 
                                          img_metas, 
                                          gt_bboxes, 
                                          gt_labels, 
                                          gt_bboxes_ignore, 
                                          proposal_cfg,
                                          **kwargs)
            name = type(sub_head).__name__
            for key, value in loss.items():
                losses["sub{}_{}_{}".format(idx, name, key)] = value

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError

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
        for idx, (sub_head, out) in enumerate(zip(self.heads, outs)): 
            #if idx != 4:
            #    continue
            sub_bbox_list = sub_head.get_bboxes(*out, img_metas, rescale=rescale)       
            for bbox, label in sub_bbox_list:
                bboxes.append(bbox)
                labels.append(label)
        
        cat_bboxes = torch.cat(bboxes)
        cat_labels = torch.cat(labels) 

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

           
            if False:
                det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
            if True:
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
