#! /usr/bin python
python /home/xiaoyu/Desktop/attribute_classify_JD_interview/train_deepmar_resnet50.py \
    --sys_device_ids="(0,)" \
    --dataset=pa100k \
    --partition_idx=0 \
    --split=trainval \
    --test_split=test \
    --batch_size=8 \
    --resize="(224,224)" \
    --new_params_lr=0.001 \
    --finetuned_params_lr=0.001 \
    --staircase_decay_at_epochs="(50,100)" \
    --total_epochs=5 \
    --epochs_per_save=1 \
    --drop_pool5=True \
    --drop_pool5_rate=0.5 \
    --resume=False \
    --ckpt_file= \
    --model_weight_file= \
