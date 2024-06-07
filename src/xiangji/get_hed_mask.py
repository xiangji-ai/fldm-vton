import base64
import io
import os
import time

import cv2
import numpy as np
import requests
from PIL import Image


def read_img_as_base64(filepath):
    with open(filepath, 'rb') as infile:
        base64_content = infile.read()
    img = Image.open(io.BytesIO(base64_content))
    img = np.array(img)
    cv2.imshow("img", img)
    img_format = "JPEG"
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_format = "PNG"
    img = Image.fromarray(img)
    with io.BytesIO() as output:
        img.save(output, format=img_format, quality=95)
        content = output.getvalue()

    png_content = base64.b64encode(content).decode('utf-8')
    res_content = f"data:image/{img_format};base64,"+png_content
    # print(res_content)
    return res_content


def api_hed(url):
    api_url = "https://www.xiangjifanyi.com/image/line-drawing"  #

    # url_response = requests.get(url)
    # buffer = io.BytesIO(url_response.content)
    # image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    st = time.time()
    response = requests.post(
        api_url,
        headers={
            "Content-Type": "application/json"
        },
        json={
            "algorithm": "hed",
            "image": url,
            "output_type": "base64"

        }
    )
    et = time.time()

    if response.ok:
        response_json = response.json()
        image = response_json["image"]
        return image

    else:
        print(response.json())
    pass


def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img


def hed_op(pic_path, mask_path, out_path, box_path):
    # 输入为：平铺cloth + 平铺cloth-mask
    pic_url = read_img_as_base64(pic_path)

    img_mask = cv2.imread(mask_path, 0)

    # 平铺cloth获取hed，并转为灰度图
    base_hed = api_hed(pic_url)
    img_hed = base64_to_image(base_hed.split(',')[1])
    img_hed = cv2.cvtColor(img_hed, cv2.COLOR_BGR2GRAY)

    # 定义腐蚀操作的结构元素
    kernel = np.ones((20, 20), np.uint8)
    # 对平铺cloth-mask进行腐蚀操作，主要是去除边缘的hed mask
    eroded_mask = cv2.erode(img_mask, kernel, iterations=1)

    # 去除掉边缘hed mask
    img_hed_inner = cv2.bitwise_and(eroded_mask, img_hed)
    ret, thresh = cv2.threshold(img_hed_inner, 127, 255, cv2.THRESH_BINARY)

    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 进行腐蚀操作为了消除噪音（也可以不做），膨胀操作为了加大保留区域
    img_hed_inner_erode = cv2.erode(thresh, kernel1, iterations=1)
    # img_hed_inner_erode = cv2.erode(img_hed_inner, kernel2, iterations=1)
    img_hed_inner_erode_dilation = cv2.dilate(img_hed_inner_erode, kernel2, iterations=2)

    img_hed_bounding_box = img_hed_inner_erode_dilation.copy()

    outs = cv2.findContours(thresh, 1, 2)
    if len(outs) == 3:
        img, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    for index in range(len(contours)):
        contour = contours[index]
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area <= 100:
            cv2.rectangle(img_hed_bounding_box, (x, y), (x + w, y + h), (0, 0, 0), -1)
            continue

        cv2.rectangle(img_hed_bounding_box, (x, y), (x + w, y + h), (255, 255, 255), -1)

    cv2.imwrite(out_path, img_hed_inner_erode_dilation)

    img_hed_bounding_box = cv2.bitwise_and(eroded_mask, img_hed_bounding_box)
    cv2.imwrite(box_path, img_hed_bounding_box)


if __name__=="__main__":
    data_path = "/Users/leo/work/象寄/9.技术文档/模型训练/xj_result2/test"
    cloth_path = f"{data_path}/cloth"
    mask_path = f"{data_path}/cloth-mask"
    out_path = f"{data_path}/margin-mask"
    box_path = f"{data_path}/box-mask"
    for filename in os.listdir(cloth_path):
        if filename.endswith("jpg"):
            hed_op(f"{cloth_path}/{filename}", f"{mask_path}/{filename}", f"{out_path}/{filename}", f"{box_path}/{filename}")
