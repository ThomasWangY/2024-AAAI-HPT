for dataset in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101 imagenet
do
    bash scripts/hpt_plus/b2n_train.sh ${dataset}
    bash scripts/hpt_plus/b2n_test.sh ${dataset}
done