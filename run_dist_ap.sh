export TORCH_HOME=/youtu/xlab-team4/share/pretrained

# only need set your config
#CONFIG=configs/_sdk/faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_roix2.py
#LOGS=../logs/_sdk-w1-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_roix2.py
#OUTPUT=../logs/debug-camera

#CONFIG=configs/_sdk/faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_semi_2ndstage.py
#LOGS=../logs/_sdk-w2-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_semi_2ndstage.py
#OUTPUT=../logs/debug-camera

#CONFIG=configs/_algo/tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py
#LOGS=../logs/_algo-w1-mhnms-tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py
#OUTPUT=../logs/debug-tanners

#CONFIG=configs/_baseline/hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.py
#LOGS=../logs-camera/rpf-baseline-worker2-hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.yaml/
#OUTPUT=../logs-camera/rpf-baseline-worker2-hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.yaml/

#LOGS=/youtu/xlab-team4/choasliu/research/logs/_sdk_baseline_mumu/
#OUTPUT=/youtu/xlab-team4/choasliu/research/logs/_sdk_baseline_mumu/


CONFIG=configs/_camera/res2net_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.py
LOGS=../logs-camera/rpf-baseline-worker2-res2net_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.yaml/
OUTPUT=$LOGS

CHECKPOINT=$LOGS/epoch_19.pth

#python3 tools/test.py $CONFIG $CHECKPOINT \
#    --show-dir $OUTPUT

#sh tools/dist_test.sh $CONFIG $CHECKPOINT \
#    8 --show-dir $OUTPUT

sh tools/dist_test.sh $CONFIG $CHECKPOINT \
    8 --out $LOGS/results.pkl --eval bbox --options "classwise=True" 
