python -m torch.distributed.launch --nproc_per_node=4 src/train.py --multi --batch_size 16 2>&1 | tee dist.log 