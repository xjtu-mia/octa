import argparse
import os
import numpy as np
import sys
# control commands

parser = argparse.ArgumentParser()
parser.add_argument('--train_cnn', type=int, default=0,
                    help="training a Unet model for AV segmentation")
parser.add_argument('--test_cnn', type=int, default=0,
                    help="testing a Unet model for AV segmentation")
parser.add_argument('--build_graph', type=int, default=0,
                    help="building the connection graph based on CNN segmentation results")
parser.add_argument('--train_gat', type=int, default=0,
                    help="training a GAT model on graph data")
parser.add_argument('--test_gat', type=int, default=1,
                    help="testing a GAT model on graph data")
parser.add_argument('--extract_result', type=int, default=1,
                    help="extract GAT and CNN results to a new folder for post-process")
parser.add_argument("--name", type=str, default='GAT_final_66',
                        help="name of the folder for saving final result")
parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")

# training and testing the CNN model for AV segmentation
parser.add_argument('--cnn_batch_size', type=int, default=4,
                    help="batch size used for training, validation and test")
parser.add_argument('--cnn_lr', type=int, default=0.01,
                    help="initial_learning_rate")
parser.add_argument('--cnn_epochs', type=int, default=30,
                    help="number of training epochs")
parser.add_argument('--num_workers', type=int, default=4,
                    help="number of dataloader thread")
parser.add_argument('--pin_memory', type=int, default=4,
                    help="pin_memory")
parser.add_argument('--Data_path', type=str, default=sys.path[0]+'/datasets/Data66/OCTA/Data_fold_1',
                    help="OCTA Data_path ")
parser.add_argument('--cnn_Result_path', type=str, default=sys.path[0] + '/Core/UNet/Result',
                    help="OCTA Result_path ")
parser.add_argument('--test_all', type=int, default=1,
                        help="0: testing on test set, 1: testing on all images")

# building vessel graph based on results of CNN model
parser.add_argument("--contour", type=int, default=9,
                        help="Thichness of the contour")
parser.add_argument("--octa_path", type=str, default=sys.path[0]+'/datasets/Data66/OCTA/data_Fusion/*.jpg',
                    help="the images fused by 4 depth OCTA image")
parser.add_argument("--seg_path", type=str, default=sys.path[0]+'/Core/UNet/Result/Test_Predict_all/*seg.tif',
                    help="the binary segmentation results of CNN model")
parser.add_argument("--seg_gt_path", type=str, default=sys.path[0]+'/datasets/Data66/OCTA/Data_fold_1/test_all_GT/*.tif',
                    help="segmentation ground truth")
parser.add_argument("--seg_vessel_path", type=str, default=sys.path[0]+'/Core/UNet/Result/Test_Predict_all/*seg_vessel_prob.tif',
                    help="segmentation probability map")
parser.add_argument("--prob_path", type=str, default=sys.path[0]+'/Core/UNet/Result/Test_Predict_all/*Prob.npy',
                    help="the .npy of the prediction of CNN model")
parser.add_argument("--graph_save_path", type=str, default=sys.path[0]+'/Core/GAT/Buildgraph',
                        help="graph save path")

# training and testing the GAT model for AV segmentation
parser.add_argument('--gat_Result_path', type=str, default=sys.path[0] + '/Core/GAT',
                    help="OCTA Result_path ")
parser.add_argument("--gat_epochs", type=int, default=100,
                        help="number of training epochs")
parser.add_argument("--num-heads", type=int, default=4,
                    help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=6,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=4,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=256,
                    help="number of hidden units")  # 可以调整小试试
parser.add_argument("--residual", action="store_true", default=True,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=0,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0,
                    help="attention dropout")
parser.add_argument("--gat_lr", type=float, default=0.005,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=0,
                    help="weight decay")
parser.add_argument('--alpha', type=float, default=0.2,
                    help="the negative slop of leaky relu")
parser.add_argument('--gat_batch_size', type=int, default=4,
                    help="batch size used for training, validation and test")
parser.add_argument('--patience', type=int, default=10,
                    help="used for early stop")
args = parser.parse_args()
def run(args):
    if args.train_cnn:
        from Core.UNet.train_valid import main as train_CNN
        train_CNN(args)
    if args.test_cnn:
        from Core.UNet.test import main as test_CNN
        if args.test_all:
            test_CNN(args)
            args.test_all = 0
            test_CNN(args)
        else:
            test_CNN(args)
    if args.build_graph:
        from Core.GAT.Buildgraph.build_graph_slic import main as building_graph
        building_graph(args)
    if args.train_gat:
        from Core.GAT.train import main as train_GAT
        train_GAT(args)
    if args.test_gat:
        from Core.GAT.multi_models_test import main as test_GAT
        test_GAT(args)
    if args.extract_result:
        from extract_results import main as extractor
        extractor(args)

if __name__ == '__main__':
    run(args)
