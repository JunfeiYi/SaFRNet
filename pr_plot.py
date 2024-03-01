# =========================================================
# @purpose: plot PR curve by COCO API and mmdet API
# @date：   2020/12
# @version: v1.0
# @author： Xu Huasheng
# @github： https://github.com/xuhuasheng/mmdetection_plot_pr_curve
# =========================================================

import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset


def plot_pr_curve(config_file, result_file, class_id=1, metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """
    
    cfg = Config.fromfile(config_file)
    # turn on test mode of dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build dataset
    dataset = build_dataset(cfg.data.test)
    # load result file in pkl format
    pkl_results = mmcv.load(result_file)
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results, _ = dataset.format_results(pkl_results)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results[metric]) 
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    precisions_pr = precisions[:, :, 1:, :, :]
    mean = np.mean(precisions_pr,2,keepdims=True) 
    return mean[5, ::5, :, 0, 2]
    #return precisions[0, :, class_id, 0, 2]

def yolov7_pr():
    """
    anno_json = './02PR-Curves/YOLOv7/instances_val2017_.json'  # annotations json
    pred_json = './02PR-Curves/YOLOv7/bp.json'  # predictions json
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')           
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
            #print(eval._gts)            
    precisions = eval.eval["precision"]
    precisions_pr = precisions[:, :, 1:, :, :]
    mean = np.mean(precisions_pr,2,keepdims=True) 
    return mean[0, ::5, :, 0, 2]
    """
    return [[          1],
 [          1],
 [    0.99918],
 [    0.99918],
 [    0.99918],
 [    0.99918],
 [    0.99918],
 [    0.99828],
 [    0.99695],
 [    0.99612],
 [    0.99499],
 [    0.99443],
 [    0.99259],
 [    0.99219],
 [    0.99061],
 [    0.98893],
 [     0.9862],
 [    0.98086],
 [    0.96257],
 [    0.61931],
 [    0.20186]]
 
 
def yolov7_pr75():
    return [[          1],
 [    0.97564],
 [    0.97142],
 [    0.97077],
 [    0.95578],
 [    0.94742],
 [    0.94711],
 [    0.93978],
 [    0.93301],
 [    0.92591],
 [    0.91941],
 [    0.90527],
 [    0.89721],
 [    0.87028],
 [    0.67197],
 [    0.57534],
 [    0.54296],
 [    0.34979],
 [    0.11207],
 [          0],
 [          0]]


def plot_curve(pr_arrays):
    
    x = np.arange(0.0, 1.01, 0.05)
    # plot PR curve
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'gray', 'pink', 'olive', 'teal', 'lime']
    i = 0
    for name, pr_array in pr_arrays.items():
        if name=='Ours':
            plt.plot(x, pr_array, marker='*', linestyle='--',color=colors[i], label=name, markersize=9)
        else:
            plt.plot(x, pr_array, marker='.', linestyle='--', color=colors[i], label=name, markersize=5)
        print(i)
        i = i+1

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()
    
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--model', required=True, nargs='+', help='network pkl')
    parser.add_argument('--model-name', required=True, nargs='+', help='network name')
    parser.add_argument('--class-id', type=int, default=3, help='class name')
    args = parser.parse_args()
    class_id = args.class_id
    pr_arrays = dict()
    for model, model_name in zip(args.model, args.model_name):
        print(f'start eval {model} ...')
        config_file = f'02PR-Curves/{model}/{model_name}.py'
        result_file = f'02PR-Curves/{model}/{model}.pkl'
        pr_arrays[model] = plot_pr_curve(config_file, result_file, class_id)
    pr_arrays['YOLOv7'] = yolov7_pr75()
    print('start plot pr curve ...')
    plot_curve(pr_arrays)

if __name__ == "__main__":
    #plot_pr_curve(config_file=CONFIG_FILE, result_file=RESULT_FILE, metric="bbox")
    main()
    

    


    

