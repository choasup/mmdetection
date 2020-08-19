export TORCH_HOME=/youtu/xlab-team4/choasliu/pretrained

# only need set your config
CONFIG=configs/_algo/fcos_r50_caffe_fpn_4x4_1x_coco.py
#CONFIG=configs/_algo/retinanet_r50_fpn_1x_coco.py

ALGO=${CONFIG#*/}
LOGS=/apdcephfs/private_choasliu/logs/mmdetection/${ALGO////-}

python3 tools/train.py $CONFIG \
    --work-dir $LOGS
