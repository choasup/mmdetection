ROOT=/youtu-xlab4/choasliu/research/

LOG_frcn=$ROOT/logs/_sdk-w1-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py/20200828_174816.log.json
LOG_frcn_roix2=$ROOT/logs/_sdk-w1-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_roix2.py/20200828_183839.log.json
LOG_frcn_w2=$ROOT/logs/_sdk-w2-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py/20200831_155646.log.json
LOG_frcn_stride2=$ROOT/logs/_sdk-w2-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_stridex2.py/20200831_231533.log.json

python3 tools/analyze_logs.py plot_curve \
    $LOG_frcn \
    $LOG_frcn_roix2 \
    $LOG_frcn_w2 \
    $LOG_frcn_stride2 \
    --keys bbox_mAP \
    --legend frcn frcn_roix2 frcn_w2 frcn_stride2 \
    --out camera_mAP.pdf

