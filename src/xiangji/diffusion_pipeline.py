import argparse
import os
import os.path as osp
from contextlib import nullcontext
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.transforms import Resize

from ldm.models.diffusion.ddim2 import DDIMSampler
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler
from ldm.models.diffusion.plms2 import PLMSSampler
from ldm.util import instantiate_from_config
from utils.basic_data_prep_diffusion import prepare_diffusion_input


def diffusion_pipeline(opt,diffusion_model,sampler,data_path,out_path,im_name,c_name,arm_line_width=90,extra_flag=False,device=torch.device("cuda")):
    # basic data prepration
    ori_inpaint_mask,im_warp_mask,cloth_warp_mask,cloth_c,parse_head,image,im_mask,image_width, image_height=prepare_diffusion_input(data_path,
                                                                                                                                     out_path,
                                                                                                       im_name,
                                                                                                       c_name,
                                                                                                       arm_line_width=arm_line_width,
                                                                                                       extra_flag=extra_flag)

    # basic run diffusion
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with diffusion_model.ema_scope():
                mask_tensor = parse_head.unsqueeze(1).cuda()
                uc = None
                if opt.scale != 1.0:
                    uc = diffusion_model.learnable_vector.repeat(im_mask.shape[0], 1, 1).cuda()
                c = diffusion_model.get_learned_conditioning(cloth_c.squeeze(1).cuda())
                if c.shape[-1] == 1024:
                    c = diffusion_model.proj_out(c)
                if len(c.shape) == 2:
                    c = c.unsqueeze(1)
                ori_image = image.cuda()

                # cloth start setting
                cloth_start = cloth_warp_mask.cuda()
                cloth_emb = diffusion_model.encode_first_stage(cloth_start)
                cloth_emb = diffusion_model.get_first_stage_encoding(cloth_emb).detach()

                ts = torch.full((1,), 999, device=device, dtype=torch.long)
                start_code = diffusion_model.q_sample(cloth_emb, ts)

                # basic encoding
                inpaint_image = im_warp_mask.cuda()
                z_inpaint = diffusion_model.encode_first_stage(inpaint_image)
                z_inpaint = diffusion_model.get_first_stage_encoding(z_inpaint).detach()
                inpaint_mask = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        ori_inpaint_mask).cuda()
                shape = [4, z_inpaint.shape[-2], z_inpaint.shape[-1]]


                # iteratively sampling
                if opt.dpm_solver:
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=z_inpaint.shape[0],
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code,
                                                         rest=torch.cat((z_inpaint,
                                                                         inpaint_mask),
                                                                        dim=1))
                elif opt.plms:
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=z_inpaint.shape[0],
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code,
                                                         rest=torch.cat((z_inpaint,
                                                                         inpaint_mask),
                                                                        dim=1))
                else:
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=z_inpaint.shape[0],
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code,
                                                         rest=torch.cat((z_inpaint,
                                                                         inpaint_mask),
                                                                        dim=1))
                # basic decoding
                x_samples_ddim = diffusion_model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_source = torch.clamp((ori_image + 1.0) / 2.0, min=0.0, max=1.0).cuda()
                x_result = x_samples_ddim * (1 - mask_tensor) + (mask_tensor) * x_source
                x_result = x_result.squeeze(0)
                print(x_result.shape)
                # save operation
                if not opt.skip_save:
                    os.makedirs(osp.join(out_path, 'result'), exist_ok=True)
                    os.makedirs(osp.join(out_path, 'inpaint-mask'), exist_ok=True)
                    if image_height <= 512 and image_width <= 512:
                            x_sample = T.Resize((image_height, image_width))(x_result)
                    else:
                            ratio = image_width * 1.0 / image_height
                            final_width = int(ratio * 512)
                            x_sample = T.Resize((512, final_width))(x_result)
                    save_x = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(save_x.astype(np.uint8))
                    img.save(osp.join(osp.join(out_path, 'result'), im_name[:-4] + '_' +
                                                  c_name[:-4] + ".png"))

                    save_inpaint_mask=(T.Resize((512, final_width))(ori_inpaint_mask).squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(osp.join(osp.join(out_path, 'inpaint-mask'), im_name[:-4] + '_' +
                                                   c_name[:-4] + ".png"), save_inpaint_mask)

    print(f"Your samples are ready and waiting for you here: \n{out_path} \n"
          f" \nEnjoy.")



def prepare_diffusion_model(opt):
    """"
    preparing for diffusion process
    """
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, osp.join(opt.ckpt,'vton.ckpt'))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    return model,sampler


def load_model_from_config(config, ckpt, verbose=False):
    """
    load model from config
    :param config: the configname
    :param ckpt: the weights of diffusion model
    :param verbose: wheather to output
    :return: the evaluated model
    """
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="?",
        help="dir to store the input",
        default="./input/"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
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
    parser.add_argument(
        "--arm_line",
        type=float,
        default=90,
        help="the line width of arm for inpainting",
    )
    parser.add_argument(
        "--extra_flag",
        action='store_true',
        help="use extra mask",
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)
    # prepration for diffusion model
    diffusion_model,sampler=prepare_diffusion_model(opt)

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

