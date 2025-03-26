import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="MMR Args")
    # ####################### general settings ########################
    parser.add_argument("--local_rank", default=int(os.environ["LOCAL_RANK"]), type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="/root/autodl-fs/output")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=5)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default='', action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--distributed", default="False",help='distribute or not')

    # ####################### model general settings ########################
    parser.add_argument("--pretrained_vit", type=str,default='/root/autodl-fs/ViT-B-16.pt') # whether use pretrained model
    parser.add_argument("--local_temperature", type=float, default=0.07, help="initial temperature value, if 0, don't use temperature")#0.07
    parser.add_argument("--temperature", type=float, default=0.1, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    # # cross modal transfomer setting
    parser.add_argument("--image_mask", type=float, default=None, help="the ratio of masked image ")
    parser.add_argument("--text_mask", type=bool, default=True, help="if or not mask the text")    
    parser.add_argument("--text_cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--query_cmt_depth", type=int, default=2, help="cross modal transformer self attn layers")#2
    parser.add_argument("--match_cmt_depth", type=int, default=2, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--text_encoder_width", type=int, default=768)
    parser.add_argument("--num_queries", type=int, default=24) #24
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")#5.0

    # ####################### loss settings ########################
    parser.add_argument("--loss_names", default='mlm+itc+cta+cpa', help="which loss to use ['mlm', 'cmpm', 'id', 'itc']")
    parser.add_argument("--itc_loss_weight", type=float, default=1.0, help="itc loss weight")
    parser.add_argument("--itm_loss_weight", type=float, default=1.0, help="itm loss weight")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--mvsc_loss_weight", type=float, default=0.5, help="mvsc loss weight")
    parser.add_argument("--cta_loss_weight", type=float, default=1.0, help="mlsc loss weight")
    
    # ####################### vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(224, 224))
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--lateral", type=str, default="swin")
    
    # ####################### text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=28996)#28996

    # ####################### solver ########################
    parser.add_argument("--optimizer", type=str, default="AdamW", help="[SGD, Adam, AdamW]")
    parser.add_argument("--lr", type=float, default=1e-5)#1e-5
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05) # 0.05
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    # ####################### scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    # ####################### dataset ########################
    parser.add_argument("--dataroot", default="/data", help="文件目录")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=108)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args
