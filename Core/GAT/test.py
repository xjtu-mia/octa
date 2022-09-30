import numpy as np
import torch
import dgl
import os
from glob import glob
from natsort import natsorted
import torch.nn.functional as F
import argparse
from sklearn.metrics import f1_score, accuracy_score
from model.gat import GAT, GATv2, SAGE
from model.dataloader import OCTADataset
from torch.utils.data import DataLoader
from time import time

def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels



def evaluate_test(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
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

    loss_fcn = torch.nn.BCEWithLogitsLoss()
    test_dataset = OCTADataset(mode='test', thickness=args.contour)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate)
    num_feats = test_dataset.features.shape[1]
    g = test_dataset.graph
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    # define the model
    model = GATv2(g,
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
    save_path = 'result_' + str(args.contour)
    os.makedirs(save_path, exist_ok=True)

    checkpoints_path = 'checkpoint_'+ str(args.contour)
    for checkpoint in os.listdir(checkpoints_path):
        if checkpoint[0] =='E':
           checkpoint = os.path.join(checkpoints_path, checkpoint) 
           break
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    test_score_list = []
    predict_test_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        feats = feats.to(device)
        labels = labels.to(device)
        predict_test, score_test, loss_test = evaluate_test(feats, model, subgraph, labels.float(), loss_fcn)
        predict_test_list.append(predict_test)
        test_score_list.append(score_test)
        # print("Test F1-Score: {:.4f}".format(np.array(test_score_list).mean()))
        # print("Test Loss: {:.4f}".format(loss_test))
        # print(predict_test)
        # print("Test acc: {:.4f}".format(score_test))
        np.save('result/result{}.npy'.format(batch), np.array(predict_test))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--contour", type=int, default=7,
                        help="thickness")
    parser.add_argument("--gpu", type=int, default = -1,
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
    print(args)
    start = time()
    main(args)
    print(time()-start)
    print('all time is:', time()-start)
    with open('test_time.txt', 'w') as f:
        f.write(str(time()-start))
    # np.save('result.npy', k)
