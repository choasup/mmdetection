export TORCH_HOME=/youtu/xlab-team4/share/pretrained

# only need set your config
#CONFIG=configs/_sdk/faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors_roix2.py
#CONFIG=configs/_algo/retinanet_r50_fpn_1x_coco.py

CONFIG=configs/_algo/tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py
LOGS=../logs/_algo-w1-tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py
OUTPUT=../logs/debug-tanners-visualization

CHECKPOINT=$LOGS/latest.pth

python3 tools/test.py $CONFIG $CHECKPOINT \
    --show-dir $OUTPUT

#sh tools/dist_test.sh $CONFIG $CHECKPOINT \
#    8 --show-dir $OUTPUT
