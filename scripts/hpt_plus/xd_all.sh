bash scripts/hpt_plus/xd_train.sh 1.0 5 3

for dataset in imagenet_a imagenet_r imagenetv2 imagenet_sketch
do
    bash scripts/hpt_plus/xd_test_dg.sh ${dataset} 1.0 5 3
done

for dataset in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    bash scripts/hpt_plus/xd_test_cde.sh ${dataset} 1.0 5 3
done