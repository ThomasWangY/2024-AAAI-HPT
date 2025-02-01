# bash ./scripts/xd_train.sh
TRAINER=HPT_PLUS
CFG=xd
SHOTS=16
GPU=0

DATASET=imagenet
OUTPUT_DIR=./results
DATA=./data
ALPHA=$1
N_SET=$2
EP=$3

for SEED in 1 2 3
do

    DIR=${OUTPUT_DIR}/output_img_plus/${DATASET}/${TRAINER}/${CFG}_alpha_${ALPHA}_shots_${SHOTS}_n_${N_SET}_ep_${EP}/seed${SEED}
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
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.HPT_PLUS.ALPHA ${ALPHA} \
        TRAINER.HPT_PLUS.N_SET ${N_SET} \
        OPTIM.MAX_EPOCH ${EP}
    fi

done