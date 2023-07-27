torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 12345 \
  examples/ft_bloom_deepspeed.py \
  --model_dir checkpoints/bloom-396m-zh/ \
  --data_fn datasets/belle/train_3.5M_CN_processed.jsonl \
  --save_dir checkpoints/bloom-396m-zh-SFT-belle3.5M-deepspeed \
  --batch_size 8 \
  --ds_config scripts/ds_config_example.json \
  --ds_stage 1