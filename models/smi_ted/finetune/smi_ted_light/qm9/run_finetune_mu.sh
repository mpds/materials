python ../../finetune_regression.py \
    --n_batch 128 \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --max_len 202 \
    --d_dropout 0.1 \
    --dropout 0.1 \
    --lr_start 3e-5 \
    --lr_multiplier 1 \
    --max_epochs 500 \
    --num_feats 32 \
    --smi_ted_version 'v1' \
    --model_path '../' \
    --ckpt_filename 'smi-ted-Light_40.pt' \
    --data_root '../../moleculenet/qm9' \
    --dataset_name qm9 \
    --measure_name 'mu' \
    --checkpoints_folder './checkpoints_QM9-mu' \
    --loss_fn 'mae' \
    --target_metric 'mae' \
    --save_ckpt 1 \
    --start_seed 0 \
    --train_decoder 1 \