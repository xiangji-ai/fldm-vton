export CUDA_VISIBLE_DEVICES=5
python ../diffusion_pipeline.py \
--ckpt /mnt/miah203/chwang/xiangji \
--out_dir /mnt/miah203/chwang/xiangji/outputnew \
--data_dir /mnt/miah203/chwang/xiangji/test_2 \
--config ../xj_vton_config.yaml \
--seed 36 \
--ddim_steps 50 \
--scale 1 \
--precision full \
--dpm_solver \
--im_name 00000_00.jpg \
--c_name 10000_00.jpg \
--mask_type margin-mask \
--extra_flag