export CUDA_VISIBLE_DEVICES=5
python ../overall_pipeline.py \
--ckpt /mnt/miah203/chwang/xiangji \
--out_dir /mnt/miah203/chwang/xiangji/output_boxmask \
--data_dir /mnt/miah203/chwang/xiangji/test_2 \
--config ../xj_vton_config.yaml \
--im_name 00003_02.jpg \
--c_name 10000_00.jpg \
--arm_line 24 \
--seed 36 \
--ddim_steps 100 \
--scale 1 \
--precision full \
--dpm_solver \
--mask_type box-mask \
--extra_flag
