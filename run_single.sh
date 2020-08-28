export TORCH_HOME=/youtu/xlab-team4/share/pretrained

# only need set your config
#CONFIG=configs/_algo/tanners_r50_caffe_fpn_4x4_lr0.02_b4_1x_coco.py
#CONFIG=configs/_algo/retinanet_r50_fpn_1x_coco.py
CONFIG=configs/_sdk/faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py

ALGO=${CONFIG#*/}
LOGS=/youtu/xlab-team4/choasliu/research/logs/debug-$ALGO

python3 tools/train.py $CONFIG \
    --work-dir $LOGS

#CHECKPOINT=$LOGS/latest.pth
#python3 tools/test.py $CONFIG $CHECKPOINT \
#    --show-dir $LOGS
