from diffusion_pipeline import prepare_diffusion_model,diffusion_pipeline
from warp_pipeline import prepare_warp_model, warp_pipeline
from pytorch_lightning import seed_everything
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./output/"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="?",
        help="dir to store the input",
        default="./input/"
    )
    parser.add_argument(
        "--im_name",
        type=str,
        nargs="?",
        help="the image name",
        default="00000_00.jpg"
    )
    parser.add_argument(
        "--c_name",
        type=str,
        nargs="?",
        help="the cloth name",
        default="00000_00.jpg"
    )
    parser.add_argument(
        "--arm_line",
        type=float,
        default=90,
        help="the line width of arm for inpainting",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        nargs="?",
        help="the mask_type [box-mask|margin-mask|None]",
        default=None
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--extra_flag",
        action='store_true',
        help="use extra mask",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    # prepration for diffusion model
    warp_model = prepare_warp_model(opt.ckpt)
    diffusion_model,sampler=prepare_diffusion_model(opt)

    # run the warp process
    warp_pipeline(warp_model,
                  opt.data_dir,
                  opt.out_dir,
                  opt.im_name,
                  opt.c_name,
                  mask_type=opt.mask_type)
    print('warpping procedure has been done.')

    # run the diffusion process
    diffusion_pipeline(opt,
                       diffusion_model,
                       sampler,
                       opt.data_dir,
                       opt.out_dir,
                       opt.im_name,
                       opt.c_name,
                       arm_line_width=opt.arm_line,
                       extra_flag=opt.extra_flag,
                       device=torch.device("cuda")
                       )




    # multiple process
    # for id,item in enumerate(['00000_00.jpg','00001_02.jpg','00002_02.jpg','00003_02.jpg','00004_02.jpg']):
    #     # run the warp process
    #     warp_pipeline(warp_model,
    #                   opt.data_dir,
    #                   opt.out_dir,
    #                   item,
    #                   opt.c_name,
    #                   mask_type=opt.mask_type)
    #     print('warpping procedure has been done.')
    #     # run the diffusion process
    #     diffusion_pipeline(opt,
    #                        diffusion_model,
    #                        sampler,
    #                        opt.data_dir,
    #                        opt.out_dir,
    #                        item,
    #                        opt.c_name,
    #                        arm_line_width=opt.arm_line,
    #                        extra_flag=opt.extra_flag,
    #                        device=torch.device("cuda")
    #                        )
