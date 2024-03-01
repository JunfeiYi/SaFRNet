_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='CutOut', n_holes=(3), cutout_ratio=(0.01,0.02,0.04,0.06,0.08,0.1)),
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
classes = ( 'pinsub','pinsug','pinsumiss','plinkgrp','pnest','psusp','pvib','pvibmiss')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        img_prefix='data/coco/train2017/',
        #img_prefix='augdata/coco/train2017_rmm/',
        classes=classes,
        ann_file='data/coco/annotations/instances_train2017.json',
        #ann_file='augdata/coco/annotations/instances_train2017_rmm.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='data/coco/val2017/',
        classes=classes,
        ann_file='data/coco/annotations/instances_val2017.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='data/coco/val2017/',
        classes=classes,
        ann_file='data/coco/annotations/instances_val2017.json',
        pipeline=test_pipeline))
