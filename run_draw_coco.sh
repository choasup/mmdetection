ROOT=/youtu-xlab4/choasliu/research/mmdetection

LOG_retina_baseline=/youtu-xlab4/choasliu/research/logs/_algo-retinanet_r50_fpn_1x_coco.py/20200820_114159.log.json
LOG_fcos_baseline=/youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
LOG_reppoints_baseline=/youtu-xlab4/choasliu/research/logs/_algo-w1-reppoints_moment_r50_fpn_1x_coco.py/20200828_115326.log.json
LOG_tanners_bn_w1=/youtu-xlab4/choasliu/research/logs/_algo-w1-tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py/20200831_161556.log.json
LOG_tanners_syncbn_w1=/youtu-xlab4/choasliu/research/logs/_algo-w1-tanners-multi_r50_caffe_fpn_4x4_syncbn_lr0.02_b4_1x_coco.py/20200831_161634.log.json
LOG_tanners_fcos_w1=$ROOT/../logs/_algo-w1-tanners-fcos_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py/20200901_172028.log.json

LOG_tanners_bn_mhnms_w1=$ROOT/../logs/_algo-w1-mhnms-tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py/20200904_114425.log.json
LOG_tanners_bn_mhnms_w2=$ROOT/../logs/_algo-w2-mhnms-tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py/20200904_114720.log.json

#python3 tools/analyze_logs.py plot_curve \
#    $LOG5 \
#    --keys sub_FCOSHead_loss_cls sub_FCOSHead_loss_bbox \
#    --out losses.jpg

python3 tools/analyze_logs.py plot_curve \
    $LOG_retina_baseline \
    $LOG_fcos_baseline \
    $LOG_reppoints_baseline \
    $LOG_tanners_bn_w1 \
    $LOG_tanners_syncbn_w1 \
    $LOG_tanners_fcos_w1 \
    $LOG_tanners_bn_mhnms_w1 \
    $LOG_tanners_bn_mhnms_w2 \
    --keys bbox_mAP \
    --legend retina-w1 fcos-w1 reppoints-w1 tanners_bn_w1 tanners_syncbn_w1 tanners_fcos_w1 tanners_bn_mhnms_w1 tanners_bn_mhnms_w2 \
    --out mAP.pdf

#python3 tools/analyze_logs.py cal_train_time /youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
