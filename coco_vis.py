import os
import os.path as osp
import shutil
import cv2
import json
import numpy as np

label_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
}

color_dict = {
    0: (255, 000, 000),
    1: (255, 128, 000),
    2: (255, 255, 000),
    3: (000, 255, 000),
    4: (000, 255, 255),
    5: (000, 000, 255),
    6: (128, 000, 255),
    7: (255, 000, 255),
    8: (128, 000, 000),
    9: (000, 128, 000),
    10: (000, 000, 128)
}

BASEDIR = osp.dirname(osp.abspath(__file__))

IMGDIR = osp.join(BASEDIR)
LABDIR = osp.join(BASEDIR)
OUTDIR = osp.join(BASEDIR, 'watch')
if osp.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR)

imgnames = [name for name in os.listdir(IMGDIR) if name.split('.')[-1] in ['png', 'jpg', 'tif', 'bmp']]  # 按扩展名过滤
for imgname in imgnames:
    path_img = osp.join(IMGDIR, imgname)  
    img = cv2.imread(path_img)  # 读图
    
    path_json = osp.join(LABDIR, imgname.split('.')[0]+'.json')  # 找到对应名字的json标注文件
    with open(path_json, 'r') as fp:
        jsonData = json.load(fp)
    boxes = jsonData["shapes"]
    for box in boxes:
        cls_name = box["label"]
        xy4 = box["points"]
        xy4 = np.array(xy4,dtype=np.int0)  # 损失精度！
        # print(xy4.shape, xy4.dtype)  # shape: (4, 2) /*四个二维坐标*/, dtype: int64 /*整型*/
        # 图纸，点阵集，索引，颜色，粗细
        cv2.drawContours(img, [xy4], 0, color_dict[label_dict[cls_name]][::-1], 2)  # 画边框
    cv2.imwrite(osp.join(OUTDIR, imgname), img)
    print(imgname + ", done...")

