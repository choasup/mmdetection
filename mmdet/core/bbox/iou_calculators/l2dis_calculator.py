import torch

from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module()
class L2DisOverlaps(object):
    """2D IoU Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='l2', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return l2dis_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def l2dis_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['l2']
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        assert 0
        #lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        #rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        #wh = (rb - lt).clamp(min=0)  # [rows, 2]
        #overlap = wh[:, 0] * wh[:, 1]
        #area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        #    bboxes1[:, 3] - bboxes1[:, 1])

        #if mode == 'iou':
        #    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        #        bboxes2[:, 3] - bboxes2[:, 1])
        #    union = area1 + area2 - overlap
        #else:
        #    union = area1
    else:
        #lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        #rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        #wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        #overlap = wh[:, :, 0] * wh[:, :, 1]
        #area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        #    bboxes1[:, 3] - bboxes1[:, 1])

        #if mode == 'iou':
        #    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        #        bboxes2[:, 3] - bboxes2[:, 1])
        #    union = area1[:, None] + area2 - overlap
        #else:
        #    union = area1[:, None]

        bboxes1_center_x = (bboxes1[:, None, 0:1] + bboxes1[:, None, 2:3]) / 2 
        bboxes2_center_x = (bboxes2[:, None, 0] + bboxes2[:, None, 2]) / 2

        bboxes1_center_y = (bboxes1[:, None, 1:2] + bboxes1[:, None, 3:4]) / 2
        bboxes2_center_y = (bboxes2[:, None, 1] + bboxes2[:, None, 3]) / 2

        dis = (bboxes1_center_x - bboxes2_center_x) ** 2 + (bboxes1_center_y - bboxes2_center_y) ** 2
    
    #eps = union.new_tensor([eps])
    #union = torch.max(union, eps)
    #ious = overlap / union
    dis = dis.squeeze(-1)
    ious = 1.0 - dis / torch.max(dis)

    return ious
