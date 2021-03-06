dataset_type = 'CameraDatasetNew'

train_data_root = '/youtu/xlab-team4/share/datasets/camera/all/'
val_data_root = '/youtu/xlab-team4/share/datasets/camera/val_0824/'
test_data_root = '/youtu/xlab-team4/share/datasets/camera/val_0824/'
test_data_root = '/youtu/xlab-team4/share/datasets/ken/'
#test_data_root = '/youtu/xlab-team4/share/datasets/pesudo/'
#test_data_root = '/youtu/xlab-team4/share/datasets/pesudo/raws/20200828/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        #ann_file=train_data_root + 'annotations/train.json',
        ann_file=train_data_root + 'annotations/train_0905_filter_subclass_8.json',
        img_prefix=train_data_root + 'images/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        #ann_file=val_data_root + 'val_test.json',
        #img_prefix=val_data_root + 'images/',
        ann_file=train_data_root + 'annotations/train_0905_filter_subclass_8.json',
        img_prefix=train_data_root + 'images/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        #ann_file=test_data_root + 'val_test.json',
        #ann_file=test_data_root + 'merge_20200828-20200829-20200830.json',
        #ann_file=test_data_root + '20200828.json',
        #img_prefix=test_data_root + 'images/',
        ann_file=test_data_root + 'ken_test_cat.json',
        img_prefix=test_data_root + 'images/',
        pipeline=test_pipeline),)
evaluation = dict(interval=1, metric='bbox')
