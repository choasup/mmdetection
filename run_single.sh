export TORCH_HOME=/youtu/xlab-team4/choasliu/pretrained

$CONFIG=configs/research/faster_rcnn_r50_fpn_1x_coco.py
$LOGS=/apdcephfs/private_choasliu/logs/mmdetection/$CONFIG

python3 tools/train.py $CONFIG \
    --work-dir $LOGS
