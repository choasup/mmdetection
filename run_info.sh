python3 tools/analyze_logs.py plot_curve /youtu-xlab4/choasliu/research/logs/_algo-retinanet_r50_fpn_1x_coco.py/20200820_114159.log.json --keys loss_cls loss_bbox --out losses.pdf
python3 tools/analyze_logs.py plot_curve \
    /youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json \
    /youtu-xlab4/choasliu/research/logs/_algo-retinanet_r50_fpn_1x_coco.py/20200820_114159.log.json \
    --keys bbox_mAP \
    --legend run1 run2 \
    --out mAP.pdf
python3 tools/analyze_logs.py cal_train_time /youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
