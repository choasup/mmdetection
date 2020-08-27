LOG1=/youtu-xlab4/choasliu/research/logs/_algo-retinanet_r50_fpn_1x_coco.py/20200820_114159.log.json
LOG2=/youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json

#LOG3=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_1x_coco.py/20200825_121301.log.json
#LOG4=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_lr0.01_b4_2x_coco.py/20200826_105814.log.json
#LOG5=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_1x_lr_coco.py/20200825_144131.log.json
#LOG6=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_lr0.01_b4_1x_coco.py/20200825_202512.log.json
#LOG7=/youtu-xlab4/choasliu/research/logs/_algo-tanners_r50_caffe_fpn_4x4_lr0.02_b4_1x_coco.py/20200826_114619.log.json

LOG8=/youtu-xlab4/choasliu/research/logs/_algo-w1-tanners-fcos_r50_caffe_fpn_4x4_lr0.02_b4_1x_coco.py/20200826_203807.log.json
LOG9=/youtu-xlab4/choasliu/research/logs/_algo-w1-tanners-retina_r50_caffe_fpn_4x4_lr0.02_b4_1x_coco.py/20200826_204321.log.json
LOG10=/youtu-xlab4/choasliu/research/logs/_algo-w1-tanners_r50_caffe_fpn_4x4_lr0.02_b4_1x_coco.py/20200826_182117.log.json

#python3 tools/analyze_logs.py plot_curve \
#    $LOG5 \
#    --keys sub_FCOSHead_loss_cls sub_FCOSHead_loss_bbox \
#    --out losses.jpg

python3 tools/analyze_logs.py plot_curve \
    $LOG1 \
    $LOG2 \
    $LOG8 \
    $LOG9 \
    $LOG10 \
    --keys bbox_mAP \
    --legend retina-w1 fcos-w1 tanner-fcos tanner-retina tanner-1x-2e-2.b4-w1 \
    --out mAP.pdf

#python3 tools/analyze_logs.py cal_train_time /youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
