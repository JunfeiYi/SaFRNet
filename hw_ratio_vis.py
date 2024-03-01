# -*- coding: utf-8 -*-
# @Time    : 20-2-13 下午5:03
# @Author  : wusaifei
# @FileName: Vision_data.py
# @Software: PyCharm
import numpy
import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)


# 读取数据
ann_json = './data/coco/annotations/instances_train2017.json'
with open(ann_json) as f:
    ann=json.load(f)

#################################################################################################
#创建类别标签字典
category_dic=dict([(i['id'],i['name']) for i in ann['categories']])
counts_label=dict([(i['name'],0) for i in ann['categories']])
for i in ann['annotations']:
    counts_label[category_dic[i['category_id']]]+=1

# 标注长宽高比例
box_w = []
box_h = []
box_wh = []
box_s = []
categorys_wh = [[] for j in range(10)]
for a in ann['annotations']:
    if a['category_id'] != 0:
        box_w.append(round(a['bbox'][2],2))
        box_h.append(round(a['bbox'][3],2))
        wh = round(a['bbox'][2]/a['bbox'][3],0)
        s = a['bbox'][2] * a['bbox'][3]
        if wh <1 :
            wh = round(a['bbox'][3]/a['bbox'][2],0)
        box_wh.append(wh)
        for b in ann['images']:
            if a['image_id'] == b['id']:
                s_i = b['width'] * b['height']
                s = s / s_i
                box_s.append(s)
        categorys_wh[a['category_id']-1].append(wh)
# 所有标签的长宽高比例
box_wh_unique = list(set(box_wh))
box_wh_count=[box_wh.count(i) for i in box_wh_unique]
def sta_arr(myarr, bins):
    sta = np.arange(bins.size)
    result = []
    for i in range(0, bins.size):
        sta[i] = myarr[myarr < bins[i]].size
        str_item = ("data <" + str(bins[i]), str(round(sta[i]/myarr.size, 5)))
        result.append(str_item)
    print(result)
myarr=numpy.array(box_s)
bins=numpy.array([0.001, 0.002, 0.003, 0.003, 0.004, 0.005, 0.0060, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03,0.04])
sta_arr(myarr, bins)
# 绘图
wh_df = pd.DataFrame(box_wh_count,index=box_wh_unique,columns=['num_ratio'])
wh_df.plot(kind='bar',color="#55aacc")
plt.plot(box_s, '.')
plt.show()


