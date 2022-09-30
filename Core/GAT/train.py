import os

import numpy as np
import torch
import dgl
import torch.nn.functional as F
import argparse
from sklearn.metrics import f1_score
import sys
sys.path.append(sys.path[0]+'/Core/GAT')
from model.gat import GAT, GATv2, SAGE
from model.dataloader import OCTADataset
from torch.utils.data import DataLoader
import json
from time import time


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


def evaluate(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
        # print(labels.size(), np.argmax(labels.data.cpu().numpy(), axis=1))
        l = labels.data.cpu().numpy()
        l = np.argmax(l, axis=1)
        predict = np.argmax(predict, axis=1)
        l_n = 1 - l
        pre_n = 1 - predict
        tp = np.sum(predict * l)
        fp = np.sum(predict * l_n)
        tn = np.sum(pre_n * l_n)
        fn = np.sum(pre_n * l)

        tp_f = np.sum(l * predict)
        recall = tp_f / np.sum(l)
        specificity = tp_f / (tp_f + l_n * predict)
        f3 = tp / (tp + fn)
        # f2 = (0.5 * tp / (tp + fn) + 0.5 * tn / (tn + fp))
        f2 = recall - fp/(tn + fp)
        return score, loss_data.item(), recall, f2, f3



def main(args):
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))
    Result_path = args.gat_Result_path
    batch_size = args.gat_batch_size
    min_loss = np.inf
    # define loss function
    # pos_weight = torch.tensor([0.2, 1]).cuda()
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    train_dataset = OCTADataset(mode='train', thickness=args.contour, path=args.graph_save_path)

    valid_dataset = OCTADataset(mode='valid', thickness=args.contour, path=args.graph_save_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate)
    n_classes = 2
    print('n_classes is ', n_classes)
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
    g = g.to(device)
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    # define the model
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.alpha,
                args.residual)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gat_lr, weight_decay=args.weight_decay)
    model = model.to(device)
    for epoch in range(args.gat_epochs):
        model.train()
        loss_list = []
        for batch, data in enumerate(train_dataloader):
            subgraph, feats, labels = data
            feats = feats.to(device)
            labels = labels.to(device)
            model.g = subgraph.to(device)
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(feats.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))
        if epoch % 1 == 0:
            score_list = []
            val_loss_list = []
            sensitivity_list = []
            f2_list = []
            f3_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                feats = feats.to(device)
                labels = labels.to(device)
                subgraph = subgraph.to(device)
                score, val_loss, sensitivity, f2, f3 = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn)
                score_list.append(score)
                val_loss_list.append(val_loss)
                sensitivity_list.append(sensitivity)
                f2_list.append(f2)
                f3_list.append(f3)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            mean_sensitivity = np.array(sensitivity_list).mean()
            mean_f2 = np.array(f2_list).mean()
            mean_f3 = np.array(f3_list).mean()
            # print("Valid loss: {:.4f} ".format(mean_val_loss))
            # print("F1-Score: {:.4f} ".format(mean_score))
            # # print("sensitivity: {} ".format(sensitivity_list))
            # print("mean_sensitivity: {:.4f} ".format(mean_sensitivity))
            # print("mean_f2: {:.4f} ".format(mean_f2))
            # print("mean_f3: {:.4f} ".format(mean_f3))
            os.makedirs(Result_path+'/checkpoint_{:d}'.format(args.contour), exist_ok=True)
            if 1 - mean_f2 < min_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(min_loss, 1 - mean_f2))
                min_loss = 1 - mean_f2
                torch.save(model.state_dict(),
                           Result_path+'/checkpoint_{:d}/Epoch_{:.4f}_'.format(args.contour, min_loss) + str(epoch + 1) + '.pkl')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--gat_Result_path', type=str, default=sys.path[0] + '/Core/GAT',
                        help="OCTA Result_path ")
    parser.add_argument("--graph_save_path", type=str, default=sys.path[0] + '/Core/GAT/Buildgraph',
                        help="graph save path")
    parser.add_argument("--contour", type=int, default=9,
                        help="thickness")
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
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
    print(args)
    start = time()
    main(args)
    print('all time is:', time()-start)
    with open('time.txt', 'w') as f:
        f.write(str(time()-start))

    # np.save('result.npy', k)
