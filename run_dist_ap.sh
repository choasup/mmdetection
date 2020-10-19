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

model=${1}

CONFIG=configs/_algo/${model}.py
LOGS=../logs/rpf-algo-w1-${model}.yaml
OUTPUT=../logs/debug

CHECKPOINT=$LOGS/latest.pth

#python3 tools/test.py $CONFIG $CHECKPOINT \
#    --show-dir $OUTPUT

#sh tools/dist_test.sh $CONFIG $CHECKPOINT \
#    8 --show-dir $OUTPUT

sh tools/dist_test.sh $CONFIG $CHECKPOINT \
    8 --out $LOGS/results.pkl --eval bbox --options "classwise=True" 
