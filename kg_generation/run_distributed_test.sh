# export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4


python -m torch.distributed.launch \
    --nproc_per_node=8 main.py \
    training.batch_size=6 \
    training.fp16=False \
    data.dataset_ratio.train=0.01 \
    data.dataset_ratio.dev=0.01
    