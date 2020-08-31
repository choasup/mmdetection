LOG_retina_baseline=/youtu-xlab4/choasliu/research/logs/_algo-retinanet_r50_fpn_1x_coco.py/20200820_114159.log.json
LOG_fcos_baseline=/youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
LOG_reppoints_baseline=/youtu-xlab4/choasliu/research/logs/_algo-w1-reppoints_moment_r50_fpn_1x_coco.py/20200828_115326.log.json

#python3 tools/analyze_logs.py plot_curve \
#    $LOG5 \
#    --keys sub_FCOSHead_loss_cls sub_FCOSHead_loss_bbox \
#    --out losses.jpg

python3 tools/analyze_logs.py plot_curve \
    $LOG_retina_baseline \
    $LOG_fcos_baseline \
    $LOG_reppoints_baseline \
    --keys bbox_mAP \
    --legend retina-w1 fcos-w1 reppoints-w1 \
    --out mAP.pdf

#python3 tools/analyze_logs.py cal_train_time /youtu-xlab4/choasliu/research/logs/_algo-fcos_r50_caffe_fpn_4x4_1x_coco.py/20200820_114118.log.json
