export TORCH_HOME=/youtu/xlab-team4/share/pretrained

# only need set your config
CONFIG=configs/_sdk/faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py
#LOGS=../logs/_sdk-w1-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py
LOGS=../logs/_sdk-w2-faster_small_r50_fpn_mstrian_eeye_720p_20e_0720_7w_sdk_without_dcn_anchors.py

#CONFIG=configs/_algo/tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py
#LOGS=../logs/_algo-w1-tanners-multi_r50_caffe_fpn_4x4_bn_lr0.02_b4_1x_coco.py

#CONFIG=configs/_algo/retinanet_r50_fpn_1x_coco.py
#LOGS=../logs/_algo-retinanet_r50_fpn_1x_coco.py

#python3 tools/train.py $CONFIG \
#    --work-dir $LOGS

#CHECKPOINT=$LOGS/latest.pth
#sh tools/dist_test.sh $CONFIG $CHECKPOINT \
#    8 --out $LOGS/results.pkl --eval bbox --options "classwise=True"

CHECKPOINT=$LOGS/latest.pth
sh tools/dist_test.sh $CONFIG $CHECKPOINT \
    8 --format-only --options jsonfile_prefix=./ken-dev
