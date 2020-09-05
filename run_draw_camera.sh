ROOT=/youtu-xlab4/choasliu/research/

LOG_frcn=$ROOT/logs/_sdk-w1-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py/20200828_174816.log.json
LOG_frcn_w2=$ROOT/logs/_sdk-w2-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py/20200831_155646.log.json

LOG_frcn_roix2=$ROOT/logs/_sdk-w1-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_roix2.py/20200828_183839.log.json

LOG_frcn_fuse_semi_w1=$ROOT/logs/_sdk-w1-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_fusesemi.py/20200901_011054.log.json
LOG_frcn_fuse_semi_w2=$ROOT/logs/_sdk-w2-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_fusesemi.py/20200901_011054.log.json

LOG_frcn_semi_stage2_w2=$ROOT/logs/_sdk-w2-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_semi_2ndstage.py/20200904_150502.log.json

LOG_frcn_semi_w4=$ROOT/logs/_sdk-w4-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_semi.py/20200903_195746.log.json 

python3 tools/analyze_logs.py plot_curve \
    $LOG_frcn \
    $LOG_frcn_w2 \
    $LOG_frcn_roix2 \
    $LOG_frcn_fuse_semi_w1 \
    $LOG_frcn_fuse_semi_w2 \
    $LOG_frcn_semi_stage2_w2 \
    $LOG_frcn_semi_w4 \
    --keys bbox_mAP \
    --legend frcn frcn_w2 frcn_roix2 frcn_fusesemi_w1 frcn_fusesemi_w2 frcn_semi_stage2_w2 frcn_semi_w4 \
    --out camera_mAP.pdf

