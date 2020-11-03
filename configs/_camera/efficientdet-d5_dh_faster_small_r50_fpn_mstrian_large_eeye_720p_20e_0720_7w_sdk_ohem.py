model = dict(
    type='FasterRCNN',
    pretrained='/youtu/xlab-team4/share/pretrained/checkpoints/efficientdet-d5.pth',
    backbone=dict(
        type='EfficientDetBackbone',
        compound_coef=5,
        load_weights=True),
    neck=dict(
        type='BiFPNBackbone', 
        compound_coef=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=288,
        feat_channels=288,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', loss_weight=1.0, beta=0.1111111111111111)),
    roi_head=dict(
        type='DoubleHeadRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=288,
            featmap_strides=[4, 8, 16, 32],
            finest_scale=28),
        bbox_head=dict(
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=288,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=7,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0)),
        cls_roi_scale_factor=1.0,
        reg_roi_scale_factor=1.3))
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=128,
        max_num=128,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=100))

dataset_type = 'CameraDatasetNew'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CameraDatasetNew',
        ann_file=
        '/youtu/xlab-team4/share/datasets/xcamera/annotations/detection_train_20201019.json',
        img_prefix='/youtu/xlab-team4/share/datasets/xcamera/images/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Resize',
                img_scale=[(1333, 704), (1333, 736), (1333, 768), (1333, 800),
                           (2000, 1200),],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[109.675, 116.28, 115.53],
                std=[71.395, 59.12, 62.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=128),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=[
            'camera_gun', 'camera_round', 'camera_other', 'support_w',
            'support_overpass', 'support_dragon', 'support_h'
        ]),
    val=dict(
        type='CameraDatasetNew',
        ann_file=
        '/youtu/xlab-team4/share/datasets/xcamera/annotations/detection_val_20201019.json',
        img_prefix='/youtu/xlab-team4/share/datasets/xcamera/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 750),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[109.675, 116.28, 115.53],
                        std=[71.395, 59.12, 62.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=128),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=[
            'camera_gun', 'camera_round', 'camera_other', 'support_w',
            'support_overpass', 'support_dragon', 'support_h'
        ]),
    test=dict(
        type='CameraDatasetNew',
        ann_file=
        '/youtu/xlab-team4/share/datasets/xcamera/annotations/detection_val_20201019.json',
        img_prefix='/youtu/xlab-team4/share/datasets/xcamera/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 750),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[109.675, 116.28, 115.53],
                        std=[71.395, 59.12, 62.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=None))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
classes = [
    'camera_gun', 'camera_round', 'camera_other', 'support_w',
    'support_overpass', 'support_dragon', 'support_h'
]
find_unused_parameters = True
