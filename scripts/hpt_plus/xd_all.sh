bash scripts/hpt_plus/xd_train.sh

for dataset in imagenet_a imagenet_r imagenetv2 imagenet_sketch
do
    bash scripts/hpt_plus/xd_test_dg.sh ${dataset}
done

for dataset in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    bash scripts/hpt_plus/xd_test_cde.sh ${dataset}
done