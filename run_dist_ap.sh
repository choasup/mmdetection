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

# baseline
#CONFIG=configs/_baseline/hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.py
#LOGS=../logs-camera/rpf-baseline-worker2-hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.yaml/

#LOGS=/youtu/xlab-team4/choasliu/research/logs/_sdk_baseline_mumu/
#OUTPUT=/youtu/xlab-team4/choasliu/research/logs/_sdk_baseline_mumu/

#CONFIG=./configs/_camera/hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_clsroiscale.py
#LOGS=../logs-camera/rpf-baseline-worker2-hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_clsroiscale.yaml/

#CONFIG=./configs/_baseline/hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_coco.py
#LOGS=../logs-coco/rpf-baseline-worker2-hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_coco.yaml

#CONFIG=./configs/_camera/res2net_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_pafpn.py
#LOGS=../logs-camera/rpf-baseline-worker2-res2net_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_pafpn.yaml

# resnext101-64x4d
#CONFIG=./configs/_camera/resnext101_64x4d_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk.py
#LOGS=../logs-camera/rpf-baseline-worker2-resnext101_64x4d_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk.yaml/

# DetectorRS - mumu
#CONFIG=./configs/_baseline/detectoRS_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py
#LOGS=../logs-camera/rpf-baseline-mumu-detectorRS-dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.yaml/

# effientnet
#CONFIG=./configs/_camera/efficientdet-d0_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.py
#LOGS=../logs-camera/rpf-baseline-worker2-efficientdet-d0_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem.yaml/

#CONFIG=./configs/_camera/hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_on.py
#LOGS=../logs-camera/rpf-baseline-worker2-hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_on.yaml/

CONFIG=./configs/_camera/hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_arpn.py
LOGS=../logs-camera/rpf-baseline-worker2-hr_dh_faster_small_r50_fpn_mstrian_large_eeye_720p_20e_0720_7w_sdk_ohem_arpn.yaml/

OUTPUT=$LOGS

CHECKPOINT=$LOGS/latest.pth

#python3 tools/test.py $CONFIG $CHECKPOINT \
#    --out $LOGS/results.pkl --eval bbox --options "classwise=True"

#sh tools/dist_test.sh $CONFIG $CHECKPOINT \
#    8 --show-dir $OUTPUT

sh tools/dist_test.sh $CONFIG $CHECKPOINT \
    8 --out $LOGS/results.pkl --eval bbox --options "classwise=True" 
