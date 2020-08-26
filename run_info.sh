LOG1=/youtu-xlab4/choasliu/research/logs/_algo-retinanet_r50_fpn_1x_coco.py/20200820_114159.log.json
LOG2=/youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
LOG3=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_1x_coco.py/20200825_121301.log.json
LOG4=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_lr0.01_b4_2x_coco.py/20200826_105814.log.json
LOG5=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_1x_lr_coco.py/20200825_144131.log.json
LOG6=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_lr0.01_b4_1x_coco.py/20200825_202512.log.json
LOG7=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_lr0.02_b4_1x_coco.py/20200826_114619.log.json

#python3 tools/analyze_logs.py plot_curve \
#    $LOG5 \
#    --keys sub_FCOSHead_loss_cls sub_FCOSHead_loss_bbox \
#    --out losses.jpg

python3 tools/analyze_logs.py plot_curve \
    $LOG1 \
    $LOG2 \
    $LOG3 \
    $LOG4 \
    $LOG5 \
    $LOG6 \
    $LOG7 \
    --keys bbox_mAP \
    --legend retina fcos tanner-1x-5e-3.b4 tanner-2x-e-2.b4 tanner-1x-e-2.b8 tanner-1x-e-2.b4 tanner-1x-2e-2.b4 \
    --out mAP.pdf

python3 tools/analyze_logs.py cal_train_time /youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
