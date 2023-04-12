CURRENT_DIR=`pwd`
RANK_ID=0

TASK_NAME="hrnet"

start_time=$(date +%s)

if [ -d ${CURRENT_DIR}/log ];then
    cp -rf ${CURRENT_DIR}/log ${CURRENT_DIR}/log_${start_time}
    rm -rf ${CURRENT_DIR}/log
    mkdir -p ${CURRENT_DIR}/log
else
    mkdir -p ${CURRENT_DIR}/log
fi
echo "DEVICE_ID=$RANK_ID"

for i in {0};
do
nohup python3 tools/test.py configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py --out ./result.json --eval mAP &
done

bash run_get_log.sh $RANK_ID &


