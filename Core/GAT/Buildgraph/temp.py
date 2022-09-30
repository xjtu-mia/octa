# 使得血管确定
import os
import cv2
import json
import numpy as np
from copy import deepcopy
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import Counter
import argparse

from skimage.segmentation import slic, mark_boundaries
from skimage import io, measure

import torch
import torch.nn.functional as F
import pandas as pd

#参照老师的想法，采用镜像的策略
class BuildGraph(object):
    def __init__(self, octa_path, seg_path, gt_path, vessel_prob_path, prob_path, save_path, thickness = 1):
        """
        :param seg_path: 模型分割结果的路径
        :param gt_path: 真值路径
        :param vessel_prob_path: 血管概率图
        :param prob_path: 概率npy文件
        :param save_path: 保存的主目录
        """
        self.image_name = os.path.basename(seg_path)[:-4]
        self.octa = io.imread(octa_path)
        self.seg = io.imread(seg_path)
        self.seg_gt = io.imread(gt_path)
        self.seg_vessel = io.imread(vessel_prob_path)
        self.seg_prob = np.load(prob_path)
        self.graph_json_path = os.path.join(save_path, 'json')
        self.graph_img_path = os.path.join(save_path, 'images')
        self.graph = dict()
        self.graph['image_id'] = self.image_name
        self.graph['metric'] = {}
        self.graph['info'] = {'node_num': 0, 'link_num': 0}
        self.graph['nodes'] = list()  # [{'id':1}, {'centorid':(y,x)}, {'init_label:int'}, {'ground_truth:int'}, {'feature':array}]
        self.graph['links'] = list()
        self.ratio = 0.2  # self.ratio 是否为血管的比例 越小节点越多
        self.thickness = thickness

    def slic_sampling(self, img):
        slic_seg = slic(img, n_segments=1200, compactness=0.1, max_iter=50)
        # 可以尝试参数enforce_connectivity=False，会避免出现小区域内包含不同类别
        return slic_seg

    def draw_boundaries(self, img, sampling_result):
        color = (173 / 255, 216 / 255, 230 / 255)
        boundaries_img = mark_boundaries(img, sampling_result, color=color, mode='inner')
        boundaries_img = (boundaries_img * 255).astype(np.uint8)
        return boundaries_img

    def graph_generator(self):
        segments = self.slic_sampling(self.seg)#这里直接对分割结果进行超像素分割，不然会带来粘连
        # 超像素分割结果可视化
        graph_img = self.draw_boundaries(self.seg, segments)
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "slic.png"), graph_img)
        # print(graph_img.shape)
        graph_gt_img = self.draw_boundaries(self.seg_gt, segments)
        graph_vessel_img = self.draw_boundaries(self.seg_vessel, segments)
        graph_degree_img = self.draw_boundaries(self.seg, segments)
        # 统计区域信息
        region_label = measure.label(segments, 8) + 1
        region_prop = measure.regionprops(region_label)
        Sample_Nodes = []
        # 采样点信息提取，提取采样点的初始标签，真实标签，像素坐标等
        all_acc_num = 0
        for item, rp in enumerate(region_prop):
            node = dict()
            node['id'] = item
            node['centroid'] = (int(rp.centroid[0]), int(rp.centroid[1]))

            # 计算区域内血管类别
            init_elem = [self.seg[x, y] for x, y in rp.coords]
            init_counter = Counter(init_elem)   # 0  127  255
            init_label = max(init_counter, key=init_counter.get)
            # 转化为血管
            if init_label == 0:
                if init_counter[init_label] <= 0.75 * rp.area:  #如果判断为0但是节点内少于75%的节点为0，则修正节点类别
                    init_label = 1   # 因为 127 255均为血管所以直接设置为1表示血管
            if init_label > 0:
                init_label = 1  # 将剩余的转化为1
            node['init_label'] = int(init_label)

            # 在血管图上，判断区域是否为血管
            area_value = [self.seg_vessel[x, y] for x, y in rp.coords]
            zero_count = area_value.count(0)
            vessel_count = rp.area - zero_count
            if vessel_count/rp.area >= self.ratio:  # 影响采样点密度：节点内血管的像素点数目占比大于给定比例，节点的vessel特征设置为1
                node['vessel'] = 1
                # 用于复原图像的位置信息
                node['vessel_location'] = [(int(x), int(y), int(self.seg_vessel[x, y])) for x, y in rp.coords if self.seg_vessel[x, y] != 0]
                # max_location = rp.coords[area_value.index(max(area_value))]  # 选取最大值坐标
                # node['centroid'] = (int(max_location[0]), int(max_location[1]))
            else:
                node['vessel'] = 0

            # 计算真值
            gt_elem = [self.seg_gt[x, y] for x, y in rp.coords]
            gt_counter = Counter(gt_elem)
            gt_label = max(gt_counter, key=gt_counter.get)
            # 区域内背景像素占比小于0.8的判断为血管
            if gt_label == 0:
                if gt_counter[gt_label] <= 0.75 * rp.area:
                    gt_label = 1
            # 动静脉（127，255）转化为血管（1）
            if gt_label > 0:
                gt_label = 1
            if node['init_label'] == 1:  # 假设我们分割的出来的血管都是血管
                gt_label = 1
            node['ground_truth'] = int(gt_label)

            if node['init_label'] == node['ground_truth']:
                all_acc_num += 1
            # 面积 周长
            # node['perimeter'] = float(rp.perimeter)
            node['area'] = int(rp.area)
            # 圆度
            # node['roundness'] = 4 * np.pi * rp.area / rp.perimeter ** 2
            node['coords'] = [(xy[0], xy[1]) for xy in rp.coords.tolist()]
            node_elem = rp.coords
            y_list = node_elem[:, 0]
            x_list = node_elem[:, 1]
            #四周膨胀一个像素，这里的厚度改成半径
            contour_thickness = self.thickness
            bigcoords = list()
            for y in range(node['centroid'][0] - contour_thickness, node['centroid'][0] + contour_thickness):
                for x in range(node['centroid'][1] - contour_thickness, node['centroid'][1] + contour_thickness):
                    if (y - node['centroid'][0])**2 + (x - node['centroid'][1])**2 <= contour_thickness**2:
                        bigcoords.append([y, x])

            bigcoords = np.array(bigcoords)
            # bigcoords = bigcoords.astype(int)
            bigcoords[np.where(bigcoords < 0)] = 0
            bigcoords[np.where(bigcoords >= 320)] = 319

            bigcoords = set([(yx[0], yx[1]) for yx in bigcoords.tolist()])
            node_elem = set([(yx[0], yx[1]) for yx in node_elem.tolist()])
            contour = bigcoords - node_elem
            node['contour'] = list(contour)#每个node的外轮廓

            #四周膨胀一个像素
            #节点是背景的话还是只算1
            # if node["vessel"] != 0:
            #     contour_thicknesses = [1, self.thickness]
            # else:
            #     self.thickness = 1
            #     contour_thicknesses = [1, self.thickness]

            # for contour_thickness in contour_thicknesses:
            #     node_elem = rp.coords
            #     y_list = node_elem[:, 0]
            #     x_list = node_elem[:, 1]
            #     ymxm = node_elem - contour_thickness  # y-1 x-1
            #     ymxo = np.c_[y_list - contour_thickness, x_list]  # y-1 x
            #     ymxp = np.c_[y_list - contour_thickness, x_list + contour_thickness]  # y-1 x+1
            #     yoxm = np.c_[y_list, x_list - contour_thickness]  # y  x-1
            #     yoxp = np.c_[y_list, x_list + contour_thickness]  # y  x+1
            #     ypxm = np.c_[y_list + contour_thickness, x_list - contour_thickness]  # y+1 x-1
            #     ypxo = np.c_[y_list + contour_thickness, x_list]  # y+1  x
            #     ypxp = node_elem + contour_thickness  # y+1 x+1

            #     bigcoords = ymxm.tolist() + ymxo.tolist() + ymxp.tolist() + yoxm.tolist() \
            #                 + yoxp.tolist() + ypxm.tolist() + ypxo.tolist() + ypxp.tolist()
            #     bigcoords = np.array(bigcoords)
            #     # bigcoords = bigcoords.astype(int)
            #     bigcoords[np.where(bigcoords < 0)] = 0
            #     bigcoords[np.where(bigcoords >= 320)] = 319

            #     bigcoords = set([(yx[0], yx[1]) for yx in bigcoords.tolist()])
            #     node_elem = set([(yx[0], yx[1]) for yx in node_elem.tolist()])
            #     contour = bigcoords - node_elem
            #     if contour_thickness == 1:
            #         contour_base = list(contour)#每个node的外轮廓
            #     else:
            #         contour_expand = list(contour)#每个node的外轮廓
            #     if self.thickness == 1:
            #         break
            # if self.thickness == 1:
            #     node['contour'] = contour_base
            # else:
            #     flag = 0
            #     #节点为边界上的话也不算
            #     for xy in rp.coords.tolist():
            #         if xy[0] == 0 or xy[0] == 319 or xy[1] == 0 or xy[1] == 319:
            #             flag = 1
            #             break
            #     #这部分应该排除掉那些非端点的节点，稍后补充
            #     if flag == 1:
            #         node['contour'] = contour_base
            #     else:
            #         node['contour'] = contour_expand
            Sample_Nodes.append(node)
        chain_star = [-1 for i in range(len(Sample_Nodes))]  # 类似链式前向星
        chain_idx = 0

        # 确定采样点以及边  可以改成方法, 输入SampleNodes
        for idx, src in enumerate(Sample_Nodes):
            for dst in Sample_Nodes:
                if src['vessel'] != 0 and dst['vessel'] != 0:
                    if len(set(src['contour']) & set(dst['coords'])) != 0: #两个节点有接触的话
                        if chain_star[src['id']] == -1:  #  说明graphs['nodes']里没有该节点，然后放入
                            self.graph['nodes'].append(deepcopy(src))  # 用深拷贝，因为要修改值，而append是拷贝的引用
                            self.graph['nodes'][-1]['id'] = chain_idx
                            chain_star[src['id']] = chain_idx
                            src_id = chain_idx
                            chain_idx += 1
                            #XG1122Add
                            # print('-----------\n', self.graph['nodes'][-1]['id'])
                        else:
                            src_id = chain_star[src['id']]
                        if chain_star[dst['id']] == -1:
                            self.graph['nodes'].append(deepcopy(dst))
                            self.graph['nodes'][-1]['id'] = chain_idx
                            chain_star[dst['id']] = chain_idx
                            dst_id = chain_idx
                            chain_idx += 1
                        else:
                            dst_id = chain_star[dst['id']]

                        self.graph['links'].append({'source': src_id, 'target': dst_id})
                        # 邻接点
                        if not self.graph['nodes'][src_id].__contains__('neibor'):
                            self.graph['nodes'][src_id]['neibor'] = [src_id]  #  邻居中包含自身
                        self.graph['nodes'][src_id]['neibor'].append(dst_id)

                        vi = self.graph['nodes'][src_id]['centroid']
                        vj = self.graph['nodes'][dst_id]['centroid']

                        # 边的联系 同为血管为紫色 一个为血管另一个不为血管为黄色， 都不是血管为绿色
                        if src['init_label'] != 0 and dst['init_label'] != 0:
                            color = (255, 127, 255)
                        elif src['init_label'] != 0 or dst['init_label'] != 0:
                            color = (255, 255, 0)
                        else:
                            color = (0, 255, 0)
                        cv2.line(graph_img, (vi[1], vi[0]), (vj[1], vj[0]), color=color, thickness=1)

        # 节点个数，边个数
        self.graph['info']['node_num'] = len(self.graph['nodes'])
        self.graph['info']['link_num'] = len(self.graph['links'])
        # 计算特征 ---》 平均值 角度
        for v in self.graph['nodes']:
            # v  vertex 节点
            # 计算角度
            v_y, v_x = v['centroid']
            color = (0, 255, 0)
            neibor = sorted(v['neibor'], key=lambda neibor_id: self.graph['nodes'][neibor_id]['centroid'])
            if v['init_label'] == 1:
                color = (255, 0, 255)
                new_neibor = []
                for nei in neibor:
                    if self.graph['nodes'][nei]['init_label'] == 1:
                        new_neibor.append(nei)
                neibor = new_neibor
            else:  # 潜在血管
                vessel_neibor = [v['id']]
                for nei in neibor:
                    if self.graph['nodes'][nei]['init_label'] == 1:
                        vessel_neibor.append(nei)
                if len(vessel_neibor) > 1:
                    neibor = vessel_neibor
            # print(neibor)
            dy = self.graph['nodes'][neibor[-1]]['centroid'][0] - self.graph['nodes'][neibor[0]]['centroid'][0]
            dx = self.graph['nodes'][neibor[-1]]['centroid'][1] - self.graph['nodes'][neibor[0]]['centroid'][1]
            degree = round(np.arctan2(dy, dx), 4)
            v['degree'] = degree
            # v['features'].append(degree)
            # if degree > 7*np.pi/8:
            #     degree = degree + np.pi
            cv2.line(graph_degree_img, (v_x, v_y), (v_x + int(7 * np.cos(degree)), v_y + int(7 * np.sin(degree))),
                     color, thickness=1)


            # octa 平均值 方差
            v_coords = v['coords']
            octa_mean = np.mean([self.octa[coord[0], coord[1]] for coord in v_coords], axis=0).tolist()
            # if self.seg_vessel[coord[0], coord[1]] != 0
            octa_std = np.std([self.octa[coord[0], coord[1]] for coord in v_coords
                               if self.seg_vessel[coord[0], coord[1]] != 0], axis=0).tolist()
            v['octa_mean'] = octa_mean
            v['octa_std'] = octa_std
            # 特征图特征的平均值
            cur_prob = [self.seg_prob[coord[0], coord[1]] for coord in v_coords]
            prob_mean = np.mean(cur_prob, axis=0).tolist()
            # 标准差
            # prob_std = np.std([self.seg_prob[coord[0], coord[1]] for coord in v_coords
            #                    if self.seg_vessel[coord[0], coord[1]] != 0], axis=0).tolist()
            v['prob_mean'] = prob_mean
            # 选用  初始动静脉值 血管值  圆度 octa平均值 概率平均值
            v['features'] = v['prob_mean'] + [abs(np.cos(v['degree'])), int(v['init_label'])] # [v['t_vessel']] v['octa_mean'] + v['prob_mean'] # [node['ground_truth']]#[node['t_vessel']] + node['octa_mean']
            # [node['init_label'], node['vessel'], node['roundness']] + node['octa_mean'] + node['prob_mean']

        # 可视化 节点 节点角度 真值  可以改成方法, 输入self.graph
        for v in self.graph['nodes']:
            init_label = v['init_label']
            location = v['centroid']
            if init_label == 0:
                radius = 1
                color = (255, 255, 255)
            elif init_label == 1:
                radius = 2
                color = (255, 0, 0)

            cv2.circle(graph_img, (int(location[1]), int(location[0])), radius=radius, color=color,
                       thickness=-1)
            cv2.circle(graph_degree_img, (int(location[1]), int(location[0])), radius=radius, color=color,
                       thickness=-1)
            if init_label == 0:
                radius = 2
                color = (0, 255, 0)
            cv2.circle(graph_vessel_img, (int(location[1]), int(location[0])), radius=radius, color=color,
                       thickness=-1)

            if v['ground_truth'] == 0:
                radius = 1
                color = (255, 255, 255)
            elif v['ground_truth'] == 1:
                radius = 2
                color = (255, 0, 0)

            cv2.circle(graph_gt_img, (int(location[1]), int(location[0])), radius=radius, color=color,
                       thickness=-1)

        # 计算精度
        node_num = len(self.graph['nodes'])
        TP, TN, FP, FN = 0, 0, 0, 0
        for v in self.graph['nodes']:
            if v['init_label'] == 0:
                if v['ground_truth'] == 0:
                    TN += 1                     # init_label=ground_truth=0
                else:
                    FN += 1                     # init_label=1 ground_truth=0
            else:                               # v['init_label'] = 1
                if v['ground_truth'] == 0:
                    FP += 1                     # init_label=1 ground_truth=0
                else:
                    TP += 1                     # init_label=1 ground_truth=1

        self.graph['metric']['TP'] = TP  # {'TP': TP,'TN': TN, 'FP': FP, 'FN':FN}
        self.graph['metric']['FP'] = FP
        self.graph['metric']['TN'] = TN
        self.graph['metric']['FN'] = FN
        self.graph['metric']['Precision'] = TP / (TP + FP)
        self.graph['metric']['Recall'] = TP / (TP + FN)
        self.graph['metric']['F1'] = 2 * (self.graph['metric']['Recall'] * self.graph['metric']['Precision']) / (
                self.graph['metric']['Recall'] + self.graph['metric']['Precision'])
        self.graph['metric']['Acc'] = (TP + TN) / (TP + FP + TN + FN)
        self.graph['metric']['Average_Acc'] = (TP/(TP + FN) + TN / (FP + TN)) / 2
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "graph.png"), graph_img)
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "graph_degree.png"), graph_degree_img)
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "graph_gt.png"), graph_gt_img)
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "graph_vessel.png"), graph_vessel_img)
        # print(os.path.join(self.graph_img_path, self.image_name + "node.png"))
        # return chain_star

    def save_json(self):
        """
        保存graph到json文件
        :return:
        """
        graph_json = json.dumps(self.graph)
        with open(os.path.join(self.graph_json_path, self.image_name+'.json'), 'w') as json_file:
            json_file.write(graph_json)

    def display(self, out):
        io.imshow(out)
        io.show()


class GraphDGL(object):
    def __init__(self, mode, graph_list, dgl_save_path):
        """
        将graph信息合并,转换成dgl所需要的格式
        :param mode: 'train', 'valid', 'test'
        :param graph_list: 图的节点和边信息
        :param dgl_save_path: 保存路径 'dgl_graph_info'
        """
        self.mode = mode
        self.graph_list = graph_list
        self.mode_graph = dict()
        self.mode_graph_id = list()
        self.mode_labels = list()
        self.mode_feats = list()
        self.mode_graph_path = os.path.join(dgl_save_path, mode+'_graph.json')
        self.mode_graph_id_path = os.path.join(dgl_save_path, mode+'_graph_id.npy')
        self.mode_labels_path = os.path.join(dgl_save_path, mode+'_labels.npy')
        self.mode_feats_path = os.path.join(dgl_save_path, mode + '_features.npy')

    def dataloader(self):
        self.mode_graph['directed'] = True
        self.mode_graph['multigraph'] = False
        self.mode_graph['graph'] = {}
        self.mode_graph['nodes'] = []
        self.mode_graph['links'] = []


        for item, graph in tqdm(enumerate(self.graph_list), desc=self.mode):
            start = len(self.mode_graph['nodes'])
            # mode_graph
            for node in graph['nodes']:
                self.mode_graph['nodes'].append({'id': node['id']+start})
            for link in graph['links']:
                self.mode_graph['links'].append({'source': link['source']+start,
                                                 'target': link['target']+start})
            # mode_graph_id
            self.mode_graph_id[start:start+len(graph['nodes'])] = [item+1] * len(graph['nodes'])
            # mode_labels
            label_list = [node['ground_truth'] for node in graph['nodes']]
            label_array = np.array(label_list, dtype=np.int64)
            label_encoder = F.one_hot(torch.tensor(label_array), 2)
            self.mode_labels[start:start + len(graph['nodes'])] = np.array(label_encoder)
            # model_feats
            features_list = [node['features'] for node in graph['nodes']]
            self.mode_feats[start:start + len(graph['nodes'])] = np.array(features_list)

        mode_graph_json = json.dumps(self.mode_graph)
        with open(self.mode_graph_path, 'w') as f:
            f.write(mode_graph_json)

        np.save(self.mode_graph_id_path, np.array(self.mode_graph_id))
        np.save(self.mode_labels_path, np.array(self.mode_labels))
        np.save(self.mode_feats_path, np.array(self.mode_feats))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--contour", type=int, default = 1,
                        help="Thichness of the contour")

    octa_path = '/home/pc/xg/YangPeiWei/Source_code/Data66/Origin/OCTA/data_Fusion/*.jpg'
    seg_path = '../../UNet/Result_6466/Test_Predict_all/*seg.tif'
    seg_gt_path = '/home/pc/xg/YangPeiWei/Source_code/Data66/Origin/OCTA/Data66_fold_2/test_all_GT/*.tif'
    seg_vessel_path = '../../UNet/Result_6466/Test_Predict_all/*seg_vessel_prob.tif'
    prob_path = '../../UNet/Result_6466/Test_Predict_all/*Prob.npy'
    save_path = 'graph_info66'
    dgl_save_path = 'dgl_graph_info'

    args = parser.parse_args()

    octa_path_list = natsorted(glob(octa_path))
    print(len(octa_path_list))
    seg_path_list = natsorted(glob(seg_path))
    print(len(seg_path_list))
    seg_gt_path_list = natsorted(glob(seg_gt_path))
    print(len(seg_gt_path_list))
    seg_vessel_path_list = natsorted(glob(seg_vessel_path))
    print(len(seg_vessel_path_list))
    prob_path_list = natsorted(glob(prob_path))
    data_list = zip(octa_path_list, seg_path_list, seg_gt_path_list, seg_vessel_path_list,
                    prob_path_list)

    graph_list = []
    graph_acc = dict()
    feats_data = list()
    for octa, seg, seg_gt, seg_vessel, prob in tqdm(data_list, desc='Graph Building>>>'):
        graph = BuildGraph(octa, seg, seg_gt, seg_vessel, prob, save_path, thickness = args.contour)
        graph.graph_generator()
        graph.save_json()
        graph_list.append(graph.graph)
        acc_dict = dict()
        acc_dict['TP'] = graph.graph['metric']['TP']
        acc_dict['FP'] = graph.graph['metric']['FP']
        acc_dict['TN'] = graph.graph['metric']['TN']
        acc_dict['FN'] = graph.graph['metric']['FN']
        acc_dict['Precision'] = graph.graph['metric']['Precision']
        acc_dict['Recall'] = graph.graph['metric']['Recall']
        acc_dict['F1'] = graph.graph['metric']['F1']
        acc_dict['Acc'] = graph.graph['metric']['Acc']
        acc_dict['Average_Acc'] = graph.graph['metric']['Average_Acc']
        graph_acc[graph.graph['image_id']] = acc_dict

        # print(graph.graph['image_id'], graph.graph['graph_accuracy'], graph.graph['vessels_accuracy'], graph.graph['all_accuracy'])
        # break
        # feats_data[os.path.basename(octa)] = []
        for graph_node in graph.graph['nodes']:
            feats_data.append(graph_node['features']+[graph_node['ground_truth']])
        # break

    # acc_json = json.dumps(graph_acc)
    # with open(os.path.join(save_path, 'graph_accracy.json'), 'w') as f:
    #     f.write(acc_json)
    acc_pd = pd.DataFrame.from_dict(graph_acc)
    acc_pd.to_excel('graph_metrics.xlsx')
    feats_pd = pd.DataFrame(feats_data)
    feats_pd.to_excel('feats_data.xlsx')
    print('train_1 data building')
    train_graph_list = graph_list[:12]+graph_list[24:48]
    train = GraphDGL('train', train_graph_list, dgl_save_path + '_66')
    train.dataloader()
    valid_graph_list = graph_list[12:24]
    valid = GraphDGL('valid', valid_graph_list, dgl_save_path + '_66')
    valid.dataloader()
    test_graph_list = graph_list[0:]
    test = GraphDGL('test', test_graph_list, dgl_save_path + '_66')
    test.dataloader()
    # test









