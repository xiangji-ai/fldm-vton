import argparse
import json
import os
from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def get_coco_body25_mapping():
    #left numbers are coco format while right numbers are body25 format
    return {
        0:0,
        1:1,
        2:2,
        3:3,
        4:4,
        5:5,
        6:6,
        7:7,
        8:9,
        9:10,
        10:11,
        11:12,
        12:13,
        13:14,
        14:15,
        15:16,
        16:17,
        17:18
    }


def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024, ratio=90):
    parse_array = np.array(im_parse)
    parse_upper = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))
    parse_neck = (parse_array == 10).astype(np.float32)
    agnostic = im_parse.copy()
    im_arms = Image.new('L', (w, h), 'black')
    arms_draw = ImageDraw.Draw(im_arms)
    
    pose_mapping = get_coco_body25_mapping()
    shoulder_right = tuple(pose_data[pose_mapping[2]])
    shoulder_left = tuple(pose_data[pose_mapping[5]])
    elbow_right = tuple(pose_data[pose_mapping[3]])
    elbow_left = tuple(pose_data[pose_mapping[6]])
    wrist_right = tuple(pose_data[pose_mapping[4]])
    wrist_left = tuple(pose_data[pose_mapping[7]])
    
    ARM_LINE_WIDTH = int(ratio / 384 * w)
    print(ARM_LINE_WIDTH)
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
    parse_arm = (np.array(im_arms) / 255).astype(np.float32)
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="dataset dir")
    parser.add_argument('--output_path', type=str, help="output dir")

    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    
    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(data_path, 'image-parse-v3', parse_name))
        w=np.array(im_parse).shape[1]
        h=np.array(im_parse).shape[0]
        
        # 384:512 -> ratio = 90, which is positive correlationed with the person width / the image width.
        agnostic = get_im_parse_agnostic(im_parse, pose_data,w,h)
        
        agnostic.save(osp.join(output_path, parse_name))
