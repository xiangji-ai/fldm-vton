import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse

from utils.basic_data_prep_warp import prepare_warp_input
from warpmodels.afwm import AFWM
from warpmodels.networks import load_checkpoint


def warp_pipeline(warp_model, data_path, out_path, im_name, c_name, mask_type='None'):
    """
    Overall warpping pipeline

    :param data_path: the basic data path
    :param out_path: the basic output path
    :param im_name: the image name
    :param c_name: the cloth name
    :param mask_type: the extra mask type [box-mask|margin-mask|None] None means no prepration for extra annotions.
    :return:
    """

    # prepare all warpping data
    cloth_image, cloth_mask, parsed_agnostic_mask, dense_image, extra_mask = prepare_warp_input(
        data_path, im_name, c_name, mask_type
    )

    # unify the data shape with the batch dimension and to the gpu device
    # basic input
    cloth_image = cloth_image.unsqueeze(0).cuda()
    parsed_agnostic_mask = parsed_agnostic_mask.unsqueeze(0).cuda()
    dense_image = dense_image.unsqueeze(0).cuda()
    cloth_image_down = F.interpolate(cloth_image, size=(256, 192), mode='bilinear')
    parsed_agnostic_down = F.interpolate(parsed_agnostic_mask, size=(256, 192), mode='nearest')
    dense_image_down = F.interpolate(dense_image, size=(256, 192), mode='bilinear')

    # basic mask
    cloth_mask = cloth_mask.unsqueeze(0).unsqueeze(0).cuda()
    if mask_type != None:
        extra_mask = extra_mask.unsqueeze(0).unsqueeze(0).cuda()

    # organize the input data
    basicinput = torch.cat([parsed_agnostic_down, dense_image_down], 1)

    # run the warpping model under the resolution of [256,192]
    _, last_flow = warp_model(basicinput, cloth_image_down)

    # ajust the image/mask resolution to the original one
    N, _, iH, iW = cloth_image.size()
    last_flow = F.interpolate(last_flow, size=(iH, iW), mode='bilinear')
    warped_cloth_image = F.grid_sample(cloth_image, last_flow.permute(0, 2, 3, 1),
                                       mode='bilinear', padding_mode='border')
    warped_cloth_mask = F.grid_sample(cloth_mask, last_flow.permute(0, 2, 3, 1),
                                      mode='bilinear', padding_mode='zeros')
    warped_extra_mask = None
    if mask_type != None:
        warped_extra_mask = F.grid_sample(extra_mask, last_flow.permute(0, 2, 3, 1),
                                          mode='bilinear', padding_mode='zeros')

    # save the final results
    save_warpped(out_path, im_name, c_name, warped_cloth_image, warped_cloth_mask, warped_extra_mask, mask_type)

    print('{}-{} pair has been warped'.format(im_name[:-4], c_name))

    return None


def save_warpped(out_path, im_name, c_name, warped_cloth_image, warped_cloth_mask, warped_extra_mask, mask_type=None):
    """
    save the corresponding warped results.
    :param out_path:  the output datapath
    :param im_name:  the image name
    :param c_name: the cloth name
    :param warped_cloth_image:  the final warped clothes image
    :param warped_cloth_mask: the final warped clothes mask
    :param warped_extra_mask: the final warped extra mask
    :return: None
    """

    # basic preparation
    to_img = transforms.ToPILImage()
    save_name = im_name[:-4] + '_' + c_name[:-4] + '.jpg'
    os.makedirs(osp.join(out_path, 'cloth-warp'), exist_ok=True)
    os.makedirs(osp.join(out_path, 'cloth-warp-mask'), exist_ok=True)
    os.makedirs(osp.join(out_path, 'extra-warp-mask'), exist_ok=True)

    # save the warped cloth image
    cv_img = (warped_cloth_image.squeeze().permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    rgb = (cv_img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(osp.join(out_path, 'cloth-warp', save_name), bgr)

    # save the warped cloth mask
    warped_cloth_mask = (warped_cloth_mask.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(osp.join(out_path, 'cloth-warp-mask', save_name), warped_cloth_mask)

    # save the warped extra mask
    if mask_type != None:
        warped_extra_mask = (warped_extra_mask.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(osp.join(out_path, 'extra-warp-mask', save_name), warped_extra_mask)



def prepare_warp_model(ckpt_path):
    """
    weights prepration for warpping
    :param ckpt_path: the checkpoint path of the warpping network
    :return: the final warp model
    """
    warp_model = AFWM(16)
    load_checkpoint(warp_model, osp.join(ckpt_path,'warp_viton.pth'))
    warp_model.cuda()
    warp_model.eval()
    return warp_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
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
        "--mask_type",
        type=str,
        nargs="?",
        help="the mask_type [box-mask|margin-mask|None]",
        default=None
    )
    opt = parser.parse_args()

    # prepration for diffusion model
    warp_model = prepare_warp_model(opt.ckpt)

    # run the warp process
    warp_pipeline(warp_model,
                  opt.data_dir,
                  opt.out_dir,
                  opt.im_name,
                  opt.c_name,
                  mask_type=opt.mask_type)


