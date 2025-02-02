# bash ./scripts/hpt_plus/xd_test_cde.sh caltech101
TRAINER=HPT_PLUS
CFG=xd
SHOTS=16
GPU=0
LOADEP=3

S_DATASET=imagenet
OUTPUT_DIR=./results
DATA=./data

DATASET=$1

for SEED in 1 2 3
do

    DIR=${OUTPUT_DIR}/output_img_plus/evaluation/${TRAINER}/${CFG}_shots_${SHOTS}/${DATASET}/seed${SEED}
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
        --model-dir ${OUTPUT_DIR}/output_img_plus/${S_DATASET}/${TRAINER}/${CFG}_shots_${SHOTS}/seed${SEED} \
        --load-epoch ${LOADEP} \
        --eval-only
    fi

done