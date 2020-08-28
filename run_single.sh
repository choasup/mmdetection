export TORCH_HOME=/youtu/xlab-team4/share/pretrained

# only need set your config
CONFIG=configs/_algo/tanners_r50_caffe_fpn_4x4_lr0.02_b4_1x_coco.py
#CONFIG=configs/_algo/retinanet_r50_fpn_1x_coco.py

ALGO=${CONFIG#*/}
LOGS=/youtu/xlab-team4/choasliu/research/logs/${ALGO////-}

#python3 tools/train.py $CONFIG \
#    --work-dir $LOGS

CHECKPOINT=$LOGS/latest.pth
python3 tools/test.py $CONFIG $CHECKPOINT \
    --show-dir $LOGS
