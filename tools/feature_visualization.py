import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
# 可视化head需要修改的文件有detector/atss，densehead/dense—test-mixins sinmple-test-boxes'，atss-head-boder
import torch.nn.functional as F


def featuremap_2_heatmap1(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片,初始化为0
    for c in range(feature_map.shape[1]):   # 按通道
        heatmap+=feature_map[:1,c,:,:]      # 像素值相加[1,H,W]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    #heatmap =  np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度[H,W] heatmap.squeeze(0) -
    
    heatmap = np.maximum(heatmap.squeeze(0), 0)  #返回大于0的数[H,W]
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1,[H,W]
    #heatmaps.append(heatmap)

    return heatmap

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
    heatmap = heatmap.squeeze(0) - mean
    heatmap = np.maximum(heatmap, 0)
    minn = np.min(heatmap)
    maxx = np.max(heatmap)
    per_value = 255 / (maxx - minn)
    heatmap = heatmap * per_value

    return heatmap


def get_attention( preds, temp=1.0):
    """ preds: Bs*C*W*H """
    
    assert isinstance(preds, torch.Tensor)
    preds = preds.detach()
    
    N, C, H, W= preds.shape
    value = torch.abs(preds)
        # Bs*W*H
    fea_map = value.mean(axis=1, keepdim=True)
    S_attention = (500* F.softmax((fea_map/temp).view(N,-1), dim=1)).view(H, W)
    #S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(H, W)

        # Bs*C
    channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
    C_attention = C * F.softmax(channel_map/temp, dim=1)

    return S_attention.cpu().numpy() , C_attention.view(16, 16).cpu().numpy()

def draw_feature_map1(features, img_path='', save_dir = './work_dirs/feature_map/',name = None):
    '''
    :param features: 特征层。可以是单层，也可以是一个多层的列表
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = cv2.imread(img_path)      #读取文件路径
    i=0
    if isinstance(features,torch.Tensor):   # 如果是单层
        features = [features]       # 转为列表
    for featuremap in features:     # 循环遍历
        #heatmap = featuremap_2_heatmap1(featuremap)	#主要是这个，就是取特征层整个的求和然后平均，归一化  热力图
        img = cv2.resize(img, (1333, 800))
        heatmap, heatmap_c = get_attention(featuremap)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        #img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0])) 
        heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        heatmap0 = heatmap0.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        
        superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子

       # plt.imshow(heatmap0)  # ,cmap='gray' ，这里展示下可视化的像素值
        # plt.imshow(superimposed_img)  # ,cmap='gray'
      #  plt.close()	#关掉展示的图片
        # 下面是用opencv查看图片的
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)     #这里通过安键盘取消显示继续运行。
        # cv2.destroyAllWindows()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #print(img_path[-9:-3])
        if  img_path[-8:-3]=='1016.':
            cv2.imwrite(os.path.join(save_dir, img_path[-8:-3] + str(i) + '.png'), superimposed_img) #superimposed_img：保存的是叠加在原图上的图，也可以保存过程中其他的自己看看
            plt.imshow(heatmap_c)
            plt.show()
        #print(os.path.join(save_dir, name + str(i) + '.png'))
        i = i + 1
