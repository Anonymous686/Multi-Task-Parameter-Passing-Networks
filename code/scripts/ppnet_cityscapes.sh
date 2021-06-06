export data_dir=YOUR_DATA_DIR
export save_dir=YOUR_SAVE_DIR
export num_devices=2
export batch_size=16
export lr=0.05
export wlr=0.1
export st=10
export ct=10
export backbone=resnet
export opt=Adam
export exp_name=${backbone}_${st}_vs_${ct}_channelwise_ppnet_opt${opt}_lr${lr}


CUDA_VISIBLE_DEVICES=0,1 python train_cityscapes_ppnet.py -d $data_dir  \
-a ppnet_distributed_simple -j 8 -b $batch_size -lr $lr -wlr $wlr  --tasks sd --epochs 400 -pf 50 -n $exp_name -s 87 \
--save -st $st -ct $ct  --save_dir $save_dir --devices 0,1 \
--loss_weight 1,20 --num_seg_cls 19 --opt_type $opt --layer_wise --channel_wise --epoch_wise   --backbone $backbone 