# bash scripts/hpt_plus/xd_train.sh
TRAINER=HPT_PLUS
CFG=xd
SHOTS=16
GPU=0

DATASET=imagenet
OUTPUT_DIR=./results
DATA=./data

for SEED in 1 2 3
do

    DIR=${OUTPUT_DIR}/output_img_plus/${DATASET}/${TRAINER}/${CFG}_shots_${SHOTS}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/b2n/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi

done