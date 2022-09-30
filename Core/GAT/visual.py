import os
import cv2
import numpy as np
import json
from glob import glob
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
from skimage import io
from skimage import io, measure
import argparse
import sys
sys.path.append(sys.path[0]+'/Core/GAT')
def main(args):
    contour = args.contour
    graphInfoList = natsorted(glob(args.graph_save_path+'/graph_info66_'+str(contour)+'/json/*.json'))
    image_gt_list = natsorted(glob(args.graph_save_path+'/graph_info66_'+str(contour)+'/images/*slic.png'))
    seg_list = natsorted(glob(args.cnn_Result_path+'/Test_Predict/*seg.tif'))
    seg_rgb_list = natsorted(glob(args.cnn_Result_path+'/Test_Predict/*seg_rgb.tif'))
    seg_prob_list = natsorted(glob(args.cnn_Result_path+'/Test_Predict/*seg_mean_prob.tif'))
    save_path = args.gat_Result_path+'/result_'+str(contour)
    os.makedirs(save_path, exist_ok=True)

    graphInfoList = natsorted(graphInfoList)[:12]
    image_gt_list = natsorted(image_gt_list)[:12]
    result_list = natsorted(glob(save_path + '/result*.npy'))

    acc = dict()
    for idx, result in tqdm(enumerate(result_list)):
        filename = graphInfoList[idx]
        with open(filename, 'r') as f:
            # print(filename)
            graphOriginInfo = json.load(f)
        result_name = os.path.basename(filename)[:-5]

        data = np.load(result, allow_pickle=True)
        img_label = cv2.imread(image_gt_list[idx])
        seg = cv2.imread(seg_list[idx], cv2.IMREAD_GRAYSCALE)
        seg_src = seg.copy()#UNet分割的副本
        seg_rgb = cv2.imread((seg_rgb_list[idx]))
        seg_prob = cv2.imread((seg_prob_list[idx]))
        metric = dict()
        metric['TP'] = 0
        metric['FP'] = 0
        metric['TN'] = 0
        metric['FN'] = 0
        # print(idx, result)
        for item, label in enumerate(data):
            xy = graphOriginInfo['nodes'][item]['centroid']
            init_label = graphOriginInfo['nodes'][item]['init_label']
            location = graphOriginInfo['nodes'][item]['vessel_location']
            gt = graphOriginInfo['nodes'][item]['ground_truth']
            x = xy[0]
            y = xy[1]
            if np.argmax(label) == 0:
                if gt == 0:  # 正确分为背景  TN
                    color = (0, 255, 0)
                    radius = 2
                    metric['TN'] += 1
                else:       # 错误分为背景 假阴 FN
                    color = (255, 0, 0)
                    radius = 2
                    metric['FN'] += 1
            elif np.argmax(label) == 1:
                if gt == 1:             # TP
                    color = (0, 0, 255)
                    radius = 2
                    metric['TP'] += 1
                else:                   # FP
                    color = (255, 0, 255)
                    radius = 2
                    metric['FP'] += 1
                if init_label != 1:
                    for location_xy in location:
                        if seg_prob[location_xy[1], location_xy[0], 0] > seg_prob[location_xy[1], location_xy[0], 2]:
                            # opencv BGR 所以0代表静脉，2代表动脉
                            c = [255, (0, 0, 255)]
                        else:
                            c = [127, (255, 0, 0)]
                        cv2.circle(img_label, (location_xy[1], location_xy[0]), radius=1, color=(0, 255, 255),
                                   thickness=-1)
                        if location_xy[2] > 60:
                            cv2.circle(seg, (location_xy[1], location_xy[0]), radius=1, color=c[0],
                                       thickness=-1)
                            cv2.circle(seg_rgb, (location_xy[1], location_xy[0]), radius=1, color=c[1],
                                       thickness=-1)
            #结果的后处理 2021/11/21添加
            #根据轮廓中的前景像素点比例确定结果的灰度值

            cv2.circle(img_label, (y, x), radius=2, color=color, thickness=-1)
            # for coord in graphOriginInfo['nodes'][item]['coords']:
            #     x = coord[0]
            #     y = coord[1]
            #     cv2.circle(img_gt, (y, x), radius=1, color=color, thickness=-1)

        seg_diff = np.int16(seg) - np.int16(seg_src)
        seg_diff[seg_diff != 0] = 1
        # seg_diff[seg_diff < 1] = 0
        # if np.sum(seg_diff) > 0:
        #     # cv2.imwrite('result/'+result_name +'seg_diff.tif', seg_diff*255)
        #     region_label = measure.label(seg_diff, connectivity=2)
        #     region_prop = measure.regionprops(region_label)
        #     for item, rp in enumerate(region_prop):#一个一个连接组分的处理
        #         node_coords = [(xy[0], xy[1]) for xy in rp.coords.tolist()]
        #         node_elem = rp.coords
        #         y_list = node_elem[:, 0]
        #         x_list = node_elem[:, 1]
        #         #四周膨胀一个像素
        #         contour_thickness = 1
        #         ymxm = node_elem - contour_thickness  # y-1 x-1
        #         ymxo = np.c_[y_list - contour_thickness, x_list]  # y-1 x
        #         ymxp = np.c_[y_list - contour_thickness, x_list + contour_thickness]  # y-1 x+1
        #         yoxm = np.c_[y_list, x_list - contour_thickness]  # y  x-1
        #         yoxp = np.c_[y_list, x_list + contour_thickness]  # y  x+1
        #         ypxm = np.c_[y_list + contour_thickness, x_list - contour_thickness]  # y+1 x-1
        #         ypxo = np.c_[y_list + contour_thickness, x_list]  # y+1  x
        #         ypxp = node_elem + contour_thickness  # y+1 x+1

        #         bigcoords = ymxm.tolist() + ymxo.tolist() + ymxp.tolist() + yoxm.tolist() \
        #                     + yoxp.tolist() + ypxm.tolist() + ypxo.tolist() + ypxp.tolist()
        #         bigcoords = np.array(bigcoords)
        #         # bigcoords = bigcoords.astype(int)
        #         bigcoords[np.where(bigcoords < 0)] = 0
        #         bigcoords[np.where(bigcoords >= 320)] = 319

        #         bigcoords = set([(yx[0], yx[1]) for yx in bigcoords.tolist()])
        #         node_elem = set([(yx[0], yx[1]) for yx in node_elem.tolist()])
        #         contour = bigcoords - node_elem
        #         node_contour = list(contour)#每个node的外轮廓

        #         node_contour_values = [seg_src[yx_[0], yx_[1]] for yx_ in node_contour]
        #         a_count = node_contour_values.count(255)
        #         v_count = node_contour_values.count(127)
        #         #凭空产生的
        #         if a_count == 0 and v_count == 0:
        #             for xy in rp.coords.tolist():
        #                 seg[xy[0], xy[1]] = 0
        #         elif a_count > v_count:
        #             for xy in rp.coords.tolist():
        #                 seg[xy[0], xy[1]] = 255
        #         else:
        #             for xy in rp.coords.tolist():
        #                 seg[xy[0], xy[1]] = 127

        # acc[result_name] = acc[result_name] / len([v for v in graphOriginInfo['nodes'] if v['ground_truth'] != 0])
        metric['Precision'] = metric['TP'] / (metric['TP'] + metric['FP'] + 1)
        metric['Recall'] = metric['TP'] / (metric['TP'] + metric['FN'])
        metric['f1'] = 2 * metric['Precision'] * metric['Recall'] / (metric['Precision'] + metric['Recall'] + 1)
        metric['accuracy'] = (metric['TP'] + metric['TN']) / (metric['TP'] + metric['TN'] + metric['FP'] + metric['FN'])
        metric['average_acc'] = (metric['TP'] / (metric['TP'] + metric['FN'])
                                 + metric['TN'] / (metric['TN'] + metric['FP']))/2
        acc[result_name] = metric
        cv2.imwrite(save_path +'/' + result_name+'.tif', img_label)
        cv2.imwrite(save_path +'/' + result_name + 'seg_gat.tif', seg)
        cv2.imwrite(save_path +'/' + result_name + 'seg_rgb_gat.tif', seg_rgb)
    # print(acc)
    # acc_pd = pd.DataFrame.from_dict(acc)
    # acc_pd.to_excel('评估.xlsx')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--gat_Result_path', type=str, default=sys.path[0] + '/Core/GAT',
                        help="OCTA Result_path ")
    parser.add_argument('--cnn_Result_path', type=str, default=sys.path[0] + '/Core/UNet/Result',
                        help="OCTA Result_path ")
    parser.add_argument("--graph_save_path", type=str, default=sys.path[0] + '/Core/GAT/Buildgraph',
                        help="graph save path")
    parser.add_argument("--contour", type=int, default=9,
                        help="thickness")
    args = parser.parse_args()
    main(args)