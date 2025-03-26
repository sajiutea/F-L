#export NCCL_DEBUG=info
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

CUDA_VISIBLE_DEVICES='0,1,2' torchrun --nproc_per_node=3 train.py --name='MMR'  --num_epoch=50 --batch_size=32
