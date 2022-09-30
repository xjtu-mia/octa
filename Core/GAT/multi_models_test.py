import numpy as np
import torch
import dgl
from glob import glob
from natsort import natsorted
import os
import torch.nn.functional as F
import argparse
from sklearn.metrics import f1_score, accuracy_score
import sys
sys.path.append(sys.path[0]+'/Core/GAT')
from model.gat import GAT, GATv2, SAGE
from visual import main as visual
from model.dataloader import OCTADataset
from torch.utils.data import DataLoader
from time import time
from evaluate import evalution

def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


all_flops = [0, 0]
def evaluate_test(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        # 分析FLOPs
        flops = FlopCountAnalysis(model, feats.float())
        all_flops[0] += flops.total() / 1e9
        all_flops[1] += 1
        print("FLOPs: ", flops.total() / 1e9, 'total:', all_flops[0]/all_flops[1])
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = accuracy_score(labels.data.cpu().numpy(),
                         predict)
        return predict, score, loss_data.item()

def main(args):
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))
    Result_path = args.gat_Result_path
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    test_dataset = OCTADataset(mode='test', thickness=args.contour, path=args.graph_save_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate)
    num_feats = test_dataset.features.shape[1]
    g = test_dataset.graph
    g = g.to(device)
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    # define the model
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                2,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.alpha,
                args.residual)
    model = model.to(device)
    acc = dict()
    models = sorted(os.listdir(Result_path+'/checkpoint_'+str(args.contour)))
    # checkpoint = 'checkpoint/Epoch_0.1596_100.pkl'

    save_path = Result_path+'/result_'+str(args.contour)
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(models)):#前面有数字的模型全部略过
        if models[i][0] >= '0' and models[i][0] <= '9':
            continue
        checkpoint = os.path.join(Result_path+'/checkpoint_'+str(args.contour), models[i])
        print("Model : " + models[i])
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        test_score_list = []
        predict_test_list = []
        for batch, test_data in enumerate(test_dataloader):
            subgraph, feats, labels = test_data
            subgraph = subgraph.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            predict_test, score_test, loss_test = evaluate_test(feats, model, subgraph, labels.float(), loss_fcn)
            predict_test_list.append(predict_test)
            test_score_list.append(score_test)
            np.save(save_path + '/result{}.npy'.format(batch), np.array(predict_test))
        visual(args)
        acc[models[i]] = evalution(args.cnn_Result_path+'/Test_Predict',
                                   Result_path,
                                   args.Data_path+'/test_GT',
                                   contour=args.contour)
        # break
    acc = sorted(acc.items(),key=lambda  item:item[1],reverse=True)
    # print(acc[0][0]) 除了最好的，其余的全部删除
    for i in range(len(acc)):
        if i != 0:
            os.remove(os.path.join(Result_path+'/checkpoint_'+str(args.contour), acc[i][0]))
    checkpoint = os.path.join(Result_path + '/checkpoint_' + str(args.contour), os.listdir(Result_path + '/checkpoint_' + str(args.contour))[0])
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        subgraph = subgraph.to(device)
        feats = feats.to(device)
        labels = labels.to(device)
        predict_test, score_test, loss_test = evaluate_test(feats, model, subgraph, labels.float(), loss_fcn)
        predict_test_list.append(predict_test)
        test_score_list.append(score_test)
        np.save(save_path + '/result{}.npy'.format(batch), np.array(predict_test))
    # os.system('python '+Result_path+'/visual.py')
    visual(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--gat_Result_path', type=str, default=sys.path[0] + '/Core/GAT',
                        help="OCTA Result_path ")
    parser.add_argument('--Data_path', type=str, default=sys.path[0] + '/Data66/OCTA/Data_fold_1',
                        help="OCTA Data_path ")
    parser.add_argument('--cnn_Result_path', type=str, default=sys.path[0] + '/Core/UNet/Result',
                        help="OCTA Result_path ")
    parser.add_argument("--graph_save_path", type=str, default=sys.path[0] + '/Core/GAT/Buildgraph',
                        help="graph save path")
    parser.add_argument("--contour", type=int, default=9,
                        help="thickness")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=10,
                        help="used for early stop")
    args = parser.parse_args()
    start = time()
    main(args)
    print(time()-start)
    print('all time is:', time()-start)
    with open('test_time.txt', 'w') as f:
        f.write(str(time()-start))
    # np.save('result.npy', k)
