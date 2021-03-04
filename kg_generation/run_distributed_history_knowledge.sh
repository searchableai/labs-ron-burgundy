# export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4


python -m torch.distributed.launch \
    --nproc_per_node=8 main.py \
    training.batch_size=6 \
    training.fp16=False \
    data.dataset_ratio.dev=0.1 \
    data.dataset_ratio.test=0.1 \
    data.dataset_dir.train="../dataset/history_knowledge_dataset/train.pkl" \
    data.dataset_dir.dev="../dataset/history_knowledge_dataset/dev.pkl" \
    data.dataset_dir.test="../dataset/history_knowledge_dataset/test.pkl" \
    training.generation.results_direction="../outputs/history_knowledge_generation.pkl"
