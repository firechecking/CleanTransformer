python examples/ft_bloom.py \
  --model_dir checkpoints/bloom-396m-zh/ \
  --data_fn datasets/belle/train_3.5M_CN_processed.jsonl  \
  --save_dir checkpoints/bloom-396m-zh-SFT-belle3.5M-onegpu \
  --batch_size 16