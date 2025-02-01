# bash ./scripts/xd_test_cde.sh caltech101
TRAINER=HPT_PLUS
CFG=xd
SHOTS=16
GPU=0

S_DATASET=imagenet
OUTPUT_DIR=./results
DATA=./data

DATASET=$1
ALPHA=$2
N_SET=$3
EP=$4

for SEED in 1 2 3
do

    DIR=${OUTPUT_DIR}/output_img_plus/evaluation/${TRAINER}/${CFG}_alpha_${ALPHA}_shots_${SHOTS}_n_${N_SET}_ep_${EP}/${DATASET}/seed${SEED}
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
        --model-dir ${OUTPUT_DIR}/output_img_plus/${S_DATASET}/${TRAINER}/${CFG}_alpha_${ALPHA}_shots_${SHOTS}_n_${N_SET}_ep_${EP}/seed${SEED} \
        --load-epoch ${EP} \
        --eval-only \
        TRAINER.HPT_PLUS.ALPHA ${ALPHA} \
        TRAINER.HPT_PLUS.N_SET ${N_SET}
    fi

done