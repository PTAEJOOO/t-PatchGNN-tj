### tPatchGNN ###
export CUDA_VISIBLE_DEVICES=1

patience=10
seed="0"
gpu=1

for sd in $seed;do
    python run_models.py \
        --dataset physionet --state 'def' --history 24 \
        --patience $patience --batch_size 100000 --lr 1e-3 \
        --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
        --te_dim 10 --node_dim 10 --hid_dim 64 \
        --outlayer Linear --seed $sd --gpu $gpu
done