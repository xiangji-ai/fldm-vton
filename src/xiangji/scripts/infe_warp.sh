export CUDA_VISIBLE_DEVICES=6
python ../warp_pipeline.py \
--ckpt /mnt/miah203/chwang/xiangji \
--out_dir /mnt/miah203/chwang/xiangji/outputnew \
--data_dir /mnt/miah203/chwang/xiangji/test_2 \
--im_name 00000_00.jpg \
--c_name 10000_00.jpg \
--mask_type box-mask
