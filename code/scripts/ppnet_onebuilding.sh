export save_dir=YOUR_SAVE_DIR
export data_dir=YOUR_DATA_DIR
export num_devices=4
export batch_size=16
export lr=0.05
export wlr=0.1
export st=10
export ct=10
export backbone=xception
export exp_name=onebuilding_${backbone}_${st}_vs_${ct}_channelwise_ppnet


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ppnet.py -d $data_dir \
-a ppnet_distributed -j 8 -b $batch_size -lr $lr -wlr $wlr  -r --tasks dnker --epochs 300 -pf 200 -n $exp_name -s 87 \
--save -st $st -ct $ct  --save_dir $save_dir --devices 0,1,2,3,3 \
--layer_wise --channel_wise --epoch_wise  --backbone $backbone