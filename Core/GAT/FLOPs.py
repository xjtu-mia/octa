import numpy as np
import torch
import dgl
import torch.nn.functional as F
import argparse
from sklearn.metrics import f1_score
from model.gat import GAT
from model.dataloader import OCTADataset
from torch.utils.data import DataLoader
from torchstat import stat
import json


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

    batch_size = args.batch_size
    min_loss = np.inf
    # define loss function
    # pos_weight = torch.tensor([0.2, 1]).cuda()
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    train_dataset = OCTADataset(mode='train')

    valid_dataset = OCTADataset(mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate)
    n_classes = 2
    print('n_classes is ', n_classes)
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
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
    stat(model, ())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
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
    print(args)

    main(args)