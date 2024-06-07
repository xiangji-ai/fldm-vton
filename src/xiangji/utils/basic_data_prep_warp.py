import os.path as osp

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def prepare_warp_input(data_path, im_name, c_name, mask_type=None):
    """
    data prepration for warpping
    :param data_path: the original data path
    :param im_name: the person image name
    :param mask_type: the mask type for extra mask [box-mask|margin-mask]
    :param c_name: the clothes image name
    :return: all preprated data for warpping
    """

    # flat clothes image & clothes mask
    cloth_image = get_cloth_image(data_path, c_name)
    cloth_mask = get_cloth_mask(data_path, c_name)

    # parsed image mask
    parsed_agnostic_mask = get_parsed_agnostic_mask(data_path, im_name)

    # densepose image
    dense_image = get_dense_image(data_path, im_name)

    # extra box|margin mask
    extra_mask = None
    if mask_type != None:
        extra_mask = get_extra_mask(data_path, mask_type, c_name)

    return cloth_image, cloth_mask, parsed_agnostic_mask, dense_image, extra_mask


def get_cloth_image(data_path,c_name):
    """
    get the flat clothes image for warpping
    :param data_path: the original data path
    :param c_name: the original clothes name
    :return: the flat clothes image
    """
    clothtransform =transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cloth = Image.open(osp.join(data_path, 'cloth', c_name)).convert('RGB')
    cloth = transforms.Resize((512, 512), interpolation=2)(cloth)
    cloth = clothtransform(cloth)
    return cloth


def get_cloth_mask(data_path,c_name):
    """
    get the clothes mask
    :param data_path: the original data path
    :param c_name: the original clothes name
    :return: the flat clothes mask
    """
    cloth_mask = Image.open(osp.join(data_path, 'cloth-mask', c_name))
    cloth_mask = transforms.Resize((512, 512), interpolation=0)(cloth_mask)
    cloth_mask = np.array(cloth_mask)
    cloth_mask = (cloth_mask >= 128).astype(np.float32)
    cloth_mask = torch.from_numpy(cloth_mask)  # [0,1]
    cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(float))
    # if len(cloth_mask.shape)!=2:
    #     cloth_mask=cloth_mask[:,:,0]
    return cloth_mask

def get_extra_mask(data_path,mask_type,c_name):
    """
    get the extra mask (box|margin)
    :param data_path: the original data path
    :param mask_type: [box-mask | margin-mask]
    :param c_name: the original clothes name
    :return: the flat extra mask (box|margin)
    """
    extra_mask = Image.open(osp.join(data_path, mask_type, c_name))
    extra_mask = transforms.Resize((512, 512), interpolation=0)(extra_mask)
    extra_mask = np.array(extra_mask)
    extra_mask = (extra_mask >= 128).astype(np.float32)
    extra_mask = torch.from_numpy(extra_mask)  # [0,1]
    extra_mask = torch.FloatTensor((extra_mask.numpy() > 0.5).astype(float))
    return extra_mask


def get_parsed_agnostic_mask(data_path,im_name):
    """
    get parsed agnostic mask (Note that one label one channel)
    :param data_path: the original data path
    :param im_name: the original image name
    :return: the parsed agnostic mask
    """
    # parse
    labels = {
        0: ['background', [0, 10]],
        1: ['hair', [1, 2]],
        2: ['face', [4, 13]],
        3: ['upper', [5, 6, 7]],
        4: ['bottom', [9, 12]],
        5: ['left_arm', [14]],
        6: ['right_arm', [15]],
        7: ['left_leg', [16]],
        8: ['right_leg', [17]],
        9: ['left_shoe', [18]],
        10: ['right_shoe', [19]],
        11: ['socks', [8]],
        12: ['noise', [3, 11]]
    }
    parse_name = im_name.replace('.jpg', '.png')
    image_parse_agnostic = Image.open(
        osp.join(data_path, 'image-parse-agnostic-v3.2', parse_name)
    )

    image_parse_agnostic = transforms.Resize((512, 512), interpolation=0)(image_parse_agnostic)
    parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
    parse_agnostic_map = torch.FloatTensor(20, 512, 512).zero_()
    parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
    new_parse_agnostic_map = torch.FloatTensor(13, 512, 512).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_agnostic_map[i] += parse_agnostic_map[label]
    return new_parse_agnostic_map

def get_dense_image(data_path,im_name):
    """
    get densepose image
    :param data_path: the original dense image
    :param im_name:
    :return:
    """

    parse_name = im_name.replace('.jpg', '.png')
    clothtransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    densepose_map = Image.open(osp.join(data_path, 'image-densepose', parse_name))
    densepose_map = transforms.Resize((512, 512), interpolation=2)(densepose_map)
    densepose_map = clothtransform(densepose_map)  # [-1,1]
    return densepose_map



