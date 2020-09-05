CONFIG=${1}
echo $CONFIG
export TORCH_HOME=/youtu/xlab-team4/share/pretrained

LOGS=/youtu/xlab-team4/choasliu/research/logs/debug

python3 tools/train.py $CONFIG --work-dir $LOGS

#sh tools/dist_train.sh $CONFIG \
#    8 --work-dir $LOGS
