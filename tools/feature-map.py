from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector
import cv2
import torch
import numpy as np


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:1, 0, :, :] * 0
    # for i in range(0, feature_map.shape[2]):
    #     for j in range(0, feature_map.shape[3]):
    #         for a in range(0, feature_map.shape[1]):
    #             if feature_map[:, a, i, j] < 0:
    #                 feature_map[:, a, i, j] = 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    mean = np.mean(heatmap)
    heatmap = heatmap - mean
    heatmap = np.maximum(heatmap, 0)
    minn = np.min(heatmap)
    maxx = np.max(heatmap)
    per_value = 255 / (maxx - minn)
    heatmap = heatmap * per_value

    return heatmap


def main():
    # config文件
    config_file = '/home/user/Documents/0Yi_work/tiny-mmdetection/00EX/00BestResults/74.7/atss_r50_fpn_1x_coco.py'
    # 训练好的模型
    checkpoint_file = '/home/user/Documents/0Yi_work/tiny-mmdetection/00EX/00BestResults/74.7/epoch_24.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 图片路径
    img_dir = '/home/user/Documents/0Yi_work/tiny-mmdetection/data/coco/val2017/JPEGImages/'
    # 检测后存放图片路径
    out_dir = '/home/user/Documents/0Yi_work/tiny-mmdetection/01feature_map_out/'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 测试集的图片名称txt
    test_path = '/home/user/Documents/0Yi_work/tiny-mmdetection/data/coco/val2017/JPEGImages/'
    test_list = os.listdir(test_path)
    #print(test_list)
    #fp = open(test_path, 'r')
    #test_list = fp.readlines()

    count = 0
    imgs = []
    for test in test_list:

        # 得到图片路径
        #test = test.replace('\n', '')
        #name = img_dir + test + '.jpg'
        name=img_dir + test
        # 得到原图大小
        img_src = cv2.imread(name)
        img_src_map = img_src / 255
        w = img_src.shape[1]
        h = img_src.shape[0]

        # 处理图像
        count += 1
        print('model is processing the {}/{} images.'.format(count, len(test_list)))
        feat = inference_detector(model, name)

        heatmap = featuremap_2_heatmap(feat)
        
        im = cv2.resize(heatmap[0], (w, h))
        im = cv2.applyColorMap(np.uint8(im), 2)
        im = np.float32(im) / 255
        final_heatmap = (im + img_src_map) / np.max((im + img_src_map))

        cv2.imwrite("{}/{}.jpg".format(out_dir, test), (final_heatmap*255))


if __name__ == '__main__':
    main()
