import json
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from ldm.utils.posemap import get_coco_body25_mapping


def prepare_diffusion_input(dataroot,outroot,im_name,c_name,arm_line_width=90,extra_flag=False):
    """
    data preparation for virtual try-on diffusion model
    :param dataroot: the original data path
    :param outroot: the output path for getting warped elements
    :param im_name: the person image name
    :param c_name: the clothes image name
    :param arm_line_width: the width of the arm line (for 00000 24, default is 90)
    :param extra_flag: wheather to offer extra mask [True|False (default)]
    :return:
           [the diffusion model input]
           inpaint_mask: the inpainting mask
           im_warp_mask: the combination of the clothes-agnostic person image and warped clothes image
           cloth_warp_mask: the warped clothes image (of note, it is not the mask)
           cloth_c: the global DINOv2 clothes image

           [the supplemented data]
           parse_head: the head parsed mask
           image: the input image
           im_mask: the clothes-agnostic person image
           image_width: the image width
           image_height: the image height
    """

    # basic image information derivation
    image = Image.open(osp.join(dataroot, 'image', im_name))
    image_width = np.array(image).shape[1]
    image_height = np.array(image).shape[0]
    image = image.resize((512,512))
    transform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)  # [-1,1]

    labels = {
        0: ['background', [0, 10]],  # 0 is background, 10 is neck
        1: ['hair', [1, 2]],  # 1 and 2 are hair
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

    # basic parsed mask prepration
    parse_name = im_name.replace('.jpg', '.png')
    im_parse = Image.open(osp.join(dataroot, 'image-parse-v3', parse_name))
    im_parse = im_parse.resize((512,512), Image.NEAREST)
    parse_array = np.array(im_parse)

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 2).astype(np.float32) + \
                 (parse_array == 4).astype(np.float32) + \
                 (parse_array == 13).astype(np.float32)

    parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 2).astype(np.float32) + \
                        (parse_array == 18).astype(np.float32) + \
                        (parse_array == 19).astype(np.float32)

    parser_mask_changeable = (parse_array == 0).astype(np.float32)

    parse_mask = (parse_array == 5).astype(np.float32) + \
                 (parse_array == 6).astype(np.float32) + \
                 (parse_array == 7).astype(np.float32)

    parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                        (parse_array == 12).astype(np.float32)  # the lower body is fixed

    parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    parse_head = torch.from_numpy(parse_head)  # [0,1]  # hair(1,2) + face(4,13)
    parser_mask_fixed = torch.from_numpy(parser_mask_fixed)  # hair (1,2) shoes (18,19) downer (9,12)
    parser_mask_changeable = torch.from_numpy(parser_mask_changeable)  # other place

    # further pose map prepration
    pose_name = im_name.replace('.jpg', '_keypoints.json')
    pose_mapping = get_coco_body25_mapping()
    im_arms = Image.new('L', (512,512))
    arms_draw = ImageDraw.Draw(im_arms)

    with open(osp.join(dataroot, 'openpose_json', pose_name), 'r') as f:
        data = json.load(f)
        data = data['people'][0]['pose_keypoints_2d']
        data = np.array(data)
        data = data.reshape((-1, 3))[:, :2]

        shoulder_right = tuple(data[pose_mapping[2]])
        shoulder_left = tuple(data[pose_mapping[5]])
        elbow_right = tuple(data[pose_mapping[3]])
        elbow_left = tuple(data[pose_mapping[6]])
        wrist_right = tuple(data[pose_mapping[4]])
        wrist_left = tuple(data[pose_mapping[7]])
        ARM_LINE_WIDTH = int(arm_line_width / 384 * image_width)
        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                arms_draw.line(
                    np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                arms_draw.line(
                    np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        else:
            arms_draw.line(np.concatenate(
                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

    # # load image-parse-agnostic
    parse_name = im_name.replace('.jpg', '.png')
    image_parse_agnostic = Image.open(
        osp.join(dataroot, 'image-parse-agnostic-v3.2', parse_name))
    image_parse_agnostic = transforms.Resize((512,512), interpolation=0)(
        image_parse_agnostic)
    parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
    parse_agnostic_map = torch.FloatTensor(20, 512,512).zero_()
    parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
    new_parse_agnostic_map = torch.FloatTensor(13,512,512).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_agnostic_map[i] += parse_agnostic_map[label]
    hands_mask = torch.sum(new_parse_agnostic_map[5:7], dim=0, keepdim=True)
    hands_mask = torch.clamp(hands_mask, min=0.0, max=1.0).bool().squeeze(0)
    parse_mask += im_arms.resize((512,512))  # now parse_mask : upper body+ arm
    parser_mask_fixed += hands_mask  # now parser_mask_fixed : hair shoes downner hand

    # delete neck and correct the parse image
    parse_head_2 = torch.clone(parse_head)
    parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
    parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                           np.logical_not(
                                                               np.array(parse_head_2, dtype=np.uint16))))

    # tune the amount of dilation here
    parse_mask = cv2.dilate(parse_mask, np.ones((8, 8), np.uint16), iterations=5)
    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)

    inpaint_mask = parse_mask_total
    inpaint_mask = inpaint_mask.unsqueeze(0)
    im_mask = image * (parse_mask_total)
    cloth_warp_mask=get_cloth_warp_mask(outroot,im_name,c_name)
    im_warp_mask = cloth_warp_mask * (1 - inpaint_mask) + im_mask

    # correct the mask region
    if extra_flag:
        extra_mask=get_extra_mask(outroot, im_name, c_name)
        inpaint_mask = inpaint_mask + extra_mask


    # cloth DINOv2 global condition
    cloth_c=get_cloth_c(dataroot, c_name)

    return (inpaint_mask.unsqueeze(0),
            im_warp_mask.unsqueeze(0),
            cloth_warp_mask.unsqueeze(0),
            cloth_c.unsqueeze(0),
            parse_head.unsqueeze(0),
            image.unsqueeze(0),
            im_mask.unsqueeze(0),
            image_width, image_height)



def get_cloth_c(dataroot,c_name):
    """
    the cloth global condition (DINO v2)
    :param dataroot: the original dataset
    :param c_name: the cloth name
    :return: the cloth condition
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cloth_mask = Image.open(osp.join(dataroot,  'cloth-mask', c_name))
    cloth_c = Image.open(osp.join(dataroot, 'cloth', c_name))
    cloth_c = cloth_c.resize((224, 224))
    cloth_c = transform(cloth_c)
    cloth_mask = cloth_mask.resize((224, 224))
    cloth_mask = transforms.ToTensor()(cloth_mask).squeeze(0)
    cloth_c = cloth_c * cloth_mask  # [-1,1]
    return cloth_c


def get_extra_mask(outroot,im_name,c_name):
    """
    get the extra mask if offered
    :param outroot: the output path
    :param im_name: the original image name
    :param c_name: the cloth name
    :return: the extra mask
    """
    w_name = im_name[:-4] + '_' + c_name[:-4] + '.jpg'
    extra_warp_mask = Image.open(osp.join(outroot, 'extra-warp-mask', w_name))
    extra_warp_mask = extra_warp_mask.resize((512,512))
    extra_warp_mask = transforms.ToTensor()(extra_warp_mask).squeeze(0)
    return extra_warp_mask


def get_cloth_warp_mask(outroot,im_name,c_name):
    """
    get the cloth warp image (of note it is not a mask)
    :param outroot: the output path
    :param im_name: the original image name
    :param c_name: the cloth name
    :return: the cloth warp image
    """
    # basic transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    w_name = im_name[:-4] + '_' + c_name[:-4] + '.jpg'

    # basic cloth_warp_image
    cloth_warp = Image.open(
        osp.join(outroot,'cloth-warp' , w_name))
    cloth_warp = cloth_warp.resize((512,512))
    cloth_warp = transform(cloth_warp)

    # prepare cloth_warp_mask
    cloth_warp_mask = Image.open(
        osp.join(outroot,  'cloth-warp-mask', w_name))
    cloth_warp_mask = cloth_warp_mask.resize((512,512))
    cloth_warp_mask = transforms.ToTensor()(cloth_warp_mask).squeeze(0)

    return cloth_warp * cloth_warp_mask