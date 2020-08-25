LOG1=/youtu-xlab4/choasliu/research/logs/_algo-retinanet_r50_fpn_1x_coco.py/20200820_114159.log.json
LOG2=/youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
LOG3=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_1x_coco.py/20200825_121301.log.json
LOG4=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_2x_coco.py/20200825_122452.log.json
LOG5=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_1x_lr_coco.py/20200825_144131.log.json

#python3 tools/analyze_logs.py plot_curve \
#    $LOG3 \
#    --keys loss_cls loss_bbox \
#    --out losses.pdf

python3 tools/analyze_logs.py plot_curve \
    $LOG1 \
    $LOG2 \
    $LOG3 \
    $LOG5 \
    --keys bbox_mAP \
    --legend retina fcos tanner1 tanner2 \
    --out mAP.pdf

python3 tools/analyze_logs.py cal_train_time /youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
