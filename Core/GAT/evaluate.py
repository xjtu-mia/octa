import os
import cv2
import numpy as np
from glob import glob
from sklearn.metrics import confusion_matrix 
from natsort import natsorted

def mIoU(confusion_matrix):
    intersection = np.diag(confusion_matrix)#交集
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)#并集
    IoU = intersection / union #交并比，即IoU
    MIoU = np.mean(IoU)#计算MIoU
    return MIoU

def compute_metric(files_gt, files_pre):
    
    met = np.zeros((len(files_gt),5))
    for idx in range(len(files_gt)):
        pre = files_pre[idx]
        gt = files_gt[idx]
        pre_img = cv2.imread(pre, 0)
        gt_img = cv2.imread(gt, 0)
        #这个不关注动静脉的类别
        pre_img[pre_img > 0] = 1
        gt_img[gt_img > 0] = 1

        cm = confusion_matrix(gt_img.reshape(gt_img.size), pre_img.reshape(pre_img.size), labels=[0, 1])
        met[idx, 0] = sum(cm[i, i] for i in range(cm.shape[0])) / np.sum(cm)#Accuracy
        Precision = np.zeros((cm.shape[0],))
        for i in range(cm.shape[0]):
            Precision[i] = cm[i, i] / np.sum(cm[:, i])
        met[idx, 1] = np.mean(Precision)#Precision
        Recall = np.zeros((cm.shape[0],))
        for i in range(cm.shape[0]):
            Recall[i] = cm[i, i] / np.sum(cm[i, :])
        met[idx, 2] = np.mean(Recall)#Recall
        met[idx, 3]= 2 * met[idx, 1] * met[idx, 2] / (met[idx, 2] + met[idx, 1])#F1
        met[idx, 4] = mIoU(cm)

    return met

def evalution(prediction_path, gat_path, gt_path, contour=7):
    gat_path += '/result_'+str(contour)
    # prediction_path = '../../UNet++/Result_64/Test_Predict'
    # gt_path = '../../../OCTA_1/U-Net/Data/test_GT'
    
    acc = dict()
    files_pre = natsorted(glob(os.path.join(prediction_path, '*seg.tif')))[:12]
    files_gt = natsorted(glob(os.path.join(gt_path, '*.tif')))[:12]
    files_pre_gat = natsorted(glob(os.path.join(gat_path, '*seg_gat.tif')))[:12]
    gat_metrics = compute_metric(files_gt, files_pre_gat)
    pre_metrics = compute_metric(files_gt, files_pre)
    gat_metrics_mean = np.mean(gat_metrics, axis= 0)
    pre_metrics_mean = np.mean(pre_metrics, axis=0)
    
    count = 0
    print('##########################################Metrics#######################################################')
    for i in range(5):
        print(str(gat_metrics_mean[i]) + ' VS ' + str(pre_metrics_mean[i]))
    if gat_metrics_mean[0] + gat_metrics_mean[2] > pre_metrics_mean[0] + pre_metrics_mean[2]:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Valid Train!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # return gat_metrics_mean[0] + gat_metrics_mean[2]
    return gat_metrics_mean[2]