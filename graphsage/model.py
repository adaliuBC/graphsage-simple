import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import pdb

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        #self.layer_num = layer_num
        #self.features = features
        #self.adj_lists = adj_lists
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        '''    
        agg = []
        enc = []
        layer_num = 2
        layer_gap = (1433-128)/(layer_num)
        agg0 = MeanAggregator(features, cuda=True)
        enc0 = Encoder(features, 1433, 128 + layer_gap*(layer_num-1), adj_lists, agg0, gcn=True, cuda=False)
        agg.append(agg0)
        enc.append(enc0)
        for i in range(1, layer_num):
            agg1 = MeanAggregator(lambda last_embeds: last_embeds.t(), cuda=False)
            agg.append(agg1)
            enc1 = Encoder(lambda last_embeds: last_embeds.t(), 128 + layer_gap*(layer_num-i), 128 + layer_gap*(layer_num-1-i), adj_lists, agg1, \
                    base_model=enc[i-1], gcn=True, cuda=False)
            enc.append(enc1)
            enc[i-1].num_samples = 5
        '''


        '''
        embeds = []
        embeds.append(self.enc(nodes))
        for i in range(1, self.layer_num):
            embeds_pos = self.enc[i](embeds[i-1])
            embeds.append(embeds_pos)'''
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        #print "scores.shape:\n", np.shape(scores)
        #print "scores:\n", scores
        #print "labels:\n", labels.squeeze()
        return self.xent(scores, labels.squeeze())

def load_citeseer():
    num_nodes = 3312
    num_feats = 3703
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("citeseer/citeseer.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("citeseer/citeseer.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    print feat_data[0]
    print labels[0]
    print adj_lists[0]
    return feat_data, labels, adj_lists

def run_citeseer():
    print "----------------citeseer----------------"
    np.random.seed(1)
    random.seed(1)
    num_nodes = 3312
    feat_data, labels, adj_lists = load_citeseer()
    features = nn.Embedding(3312, 3703)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
#    features.cuda()

    agg = []
    enc = []
    layer_num = 5
    layer_gap = (3703-128)/(layer_num)
    agg0 = MeanAggregator(features, cuda=True)
    enc0 = Encoder(features, 3703, 128 + layer_gap*(layer_num-1), adj_lists, agg0, gcn=True, cuda=False)
    agg.append(agg0)
    enc.append(enc0)
    '''
    for i in range(1, layer_num):
       agg1 = MeanAggregator(lambda nodes: enc[i-1](nodes).t(), cuda=False)
       enc1 = Encoder(lambda nodes: enc[i-1](nodes).t(), 128 + layer_gap*(layer_num-i), 128 + layer_gap*(layer_num-1-i), \
               adj_lists, agg1, base_model=enc[i-1], gcn=True, cuda=False)
       enc1.num_samples = 5
       agg.append(agg1)
       enc.append(enc1)
       
    '''
    agg1 = MeanAggregator(lambda nodes: enc[0](nodes).t(), cuda=False)  #//
    enc1 = Encoder(lambda nodes: enc[0](nodes).t(), 128+layer_gap*(layer_num-1), 128+layer_gap*(layer_num-1-1), adj_lists, agg1,
            base_model=enc[0], gcn=True, cuda=False)  #//
    agg.append(agg1)
    enc.append(enc1)
    agg2 = MeanAggregator(lambda nodes: enc[1](nodes).t(), cuda=False)  #//
    enc2 = Encoder(lambda nodes: enc[1](nodes).t(), 128+layer_gap*(layer_num-2), 128+layer_gap*(layer_num-3), adj_lists, agg1,
            base_model=enc[1], gcn=True, cuda=False)  #//
    agg.append(agg1)
    enc.append(enc1)
    agg3 = MeanAggregator(lambda nodes: enc[2](nodes).t(), cuda=False)  #//
    enc3 = Encoder(lambda nodes: enc[2](nodes).t(), enc[2].embed_dim, 1208, adj_lists, agg3,
            base_model=enc[2], gcn=True, cuda=False)  #//
    agg.append(agg3)
    enc.append(enc3)
    agg4 = MeanAggregator(lambda nodes: enc[3](nodes).t(), cuda=False)  #//
    enc4 = Encoder(lambda nodes: enc[3](nodes).t(), enc[3].embed_dim, 1168, adj_lists, agg4,
            base_model=enc[3], gcn=True, cuda=False)  #//
    agg.append(agg4)
    enc.append(enc4)
    agg5 = MeanAggregator(lambda nodes: enc4(nodes).t(), cuda=False)  #//
    enc5 = Encoder(lambda nodes: enc4(nodes).t(), enc4.embed_dim, 1128, adj_lists, agg5,
            base_model=enc4, gcn=True, cuda=False)  #//
    
    
    agg6 = MeanAggregator(lambda nodes: enc5(nodes).t(), cuda=False)  #//
    enc6 = Encoder(lambda nodes: enc5(nodes).t(), enc5.embed_dim, 1088, adj_lists, agg6,
            base_model=enc5, gcn=True, cuda=False)  #//
    agg7 = MeanAggregator(lambda nodes: enc6(nodes).t(), cuda=False)  #//
    enc7 = Encoder(lambda nodes: enc6(nodes).t(), enc6.embed_dim, 1048, adj_lists, agg7,
            base_model=enc6, gcn=True, cuda=False)  #//
    agg8 = MeanAggregator(lambda nodes: enc7(nodes).t(), cuda=False)  #//
    enc8 = Encoder(lambda nodes: enc7(nodes).t(), enc7.embed_dim, 1008, adj_lists, agg8,
            base_model=enc7, gcn=True, cuda=False)  #//
    agg9 = MeanAggregator(lambda nodes: enc8(nodes).t(), cuda=False)  #//
    enc9 = Encoder(lambda nodes: enc8(nodes).t(), enc8.embed_dim, 968, adj_lists, agg9,
            base_model=enc8, gcn=True, cuda=False)  #//
    agg10 = MeanAggregator(lambda nodes: enc9(nodes).t(), cuda=False)  #//
    enc10 = Encoder(lambda nodes: enc9(nodes).t(), enc9.embed_dim, 928, adj_lists, agg10,
            base_model=enc9, gcn=True, cuda=False)  #//
    
    agg11 = MeanAggregator(lambda nodes: enc10(nodes).t(), cuda=False)  #//
    enc11 = Encoder(lambda nodes: enc10(nodes).t(), enc10.embed_dim, 888, adj_lists, agg11,
            base_model=enc10, gcn=True, cuda=False)  #//
    agg12 = MeanAggregator(lambda nodes: enc11(nodes).t(), cuda=False)
    enc12 = Encoder(lambda nodes: enc11(nodes).t(), enc11.embed_dim, 848, adj_lists, agg12, 
            base_model=enc11, gcn=True, cuda=False)
    agg13 = MeanAggregator(lambda nodes: enc12(nodes).t(), cuda=False)  #//
    enc13 = Encoder(lambda nodes: enc12(nodes).t(), enc12.embed_dim, 808, adj_lists, agg13,
            base_model=enc12, gcn=True, cuda=False)  #//
    agg14 = MeanAggregator(lambda nodes: enc13(nodes).t(), cuda=False)  #//
    enc14 = Encoder(lambda nodes: enc13(nodes).t(), enc13.embed_dim, 768, adj_lists, agg14,
            base_model=enc13, gcn=True, cuda=False)  #//
    agg15 = MeanAggregator(lambda nodes: enc14(nodes).t(), cuda=False)  #//
    enc15 = Encoder(lambda nodes: enc14(nodes).t(), enc14.embed_dim, 728, adj_lists, agg15,
            base_model=enc14, gcn=True, cuda=False)  #//
    agg16 = MeanAggregator(lambda nodes: enc15(nodes).t(), cuda=False)  #//
    enc16 = Encoder(lambda nodes: enc15(nodes).t(), enc15.embed_dim, 688, adj_lists, agg16,
            base_model=enc15, gcn=True, cuda=False)  #//
    agg17 = MeanAggregator(lambda nodes: enc16(nodes).t(), cuda=False)  #//
    enc17 = Encoder(lambda nodes: enc16(nodes).t(), enc16.embed_dim, 648, adj_lists, agg17,
            base_model=enc16, gcn=True, cuda=False)  #//
    agg18 = MeanAggregator(lambda nodes: enc17(nodes).t(), cuda=False)  #//
    enc18 = Encoder(lambda nodes: enc17(nodes).t(), enc17.embed_dim, 608, adj_lists, agg18,
            base_model=enc17, gcn=True, cuda=False)  #//
    agg19 = MeanAggregator(lambda nodes: enc18(nodes).t(), cuda=False)  #//
    enc19 = Encoder(lambda nodes: enc18(nodes).t(), enc18.embed_dim, 568, adj_lists, agg19,
            base_model=enc18, gcn=True, cuda=False)  #//
    agg20 = MeanAggregator(lambda nodes: enc19(nodes).t(), cuda=False)  #//
    enc20 = Encoder(lambda nodes: enc19(nodes).t(), enc19.embed_dim, 528, adj_lists, agg20,
            base_model=enc19, gcn=True, cuda=False)  #//
    agg21 = MeanAggregator(lambda nodes: enc20(nodes).t(), cuda=False)  #//
    enc21 = Encoder(lambda nodes: enc20(nodes).t(), enc20.embed_dim, 488, adj_lists, agg21,
            base_model=enc20, gcn=True, cuda=False)  #//
    agg22 = MeanAggregator(lambda nodes: enc21(nodes).t(), cuda=False)
    enc22 = Encoder(lambda nodes: enc21(nodes).t(), enc21.embed_dim, 448, adj_lists, agg22, 
            base_model=enc21, gcn=True, cuda=False)
    agg23 = MeanAggregator(lambda nodes: enc22(nodes).t(), cuda=False)  #//
    enc23 = Encoder(lambda nodes: enc22(nodes).t(), enc22.embed_dim, 408, adj_lists, agg23,
            base_model=enc22, gcn=True, cuda=False)  #//
    agg24 = MeanAggregator(lambda nodes: enc23(nodes).t(), cuda=False)  #//
    enc24 = Encoder(lambda nodes: enc23(nodes).t(), enc23.embed_dim, 368, adj_lists, agg24,
            base_model=enc23, gcn=True, cuda=False)  #//
    agg25 = MeanAggregator(lambda nodes: enc24(nodes).t(), cuda=False)  #//
    enc25 = Encoder(lambda nodes: enc24(nodes).t(), enc24.embed_dim, 328, adj_lists, agg25,
            base_model=enc24, gcn=True, cuda=False)  #//
    agg26 = MeanAggregator(lambda nodes: enc25(nodes).t(), cuda=False)  #//
    enc26 = Encoder(lambda nodes: enc25(nodes).t(), enc25.embed_dim, 288, adj_lists, agg26,
            base_model=enc25, gcn=True, cuda=False)  #//
    agg27 = MeanAggregator(lambda nodes: enc26(nodes).t(), cuda=False)  #//
    enc27 = Encoder(lambda nodes: enc26(nodes).t(), enc26.embed_dim, 248, adj_lists, agg27,
            base_model=enc26, gcn=True, cuda=False)  #//
    agg28 = MeanAggregator(lambda nodes: enc27(nodes).t(), cuda=False)  #//
    enc28 = Encoder(lambda nodes: enc27(nodes).t(), enc27.embed_dim, 208, adj_lists, agg28,
            base_model=enc27, gcn=True, cuda=False)  #//
    agg29 = MeanAggregator(lambda nodes: enc28(nodes).t(), cuda=False)  #//
    enc29 = Encoder(lambda nodes: enc28(nodes).t(), enc28.embed_dim, 168, adj_lists, agg29,
            base_model=enc28, gcn=True, cuda=False)  #//
    agg30 = MeanAggregator(lambda nodes: enc29(nodes).t(), cuda=False)  #//
    enc30 = Encoder(lambda nodes: enc29(nodes).t(), enc29.embed_dim, 128, adj_lists, agg30,
            base_model=enc29, gcn=True, cuda=False)  #//  '''

    print "enc[4]:\n", enc[4]
    print "enc[3]:\n", enc[3]
    print "enc[2]:\n", enc[2]
    print "enc[1]:\n", enc[1]
    print "enc[0]:\n", enc[0]
#    enc[0].num_samples = 5
#    enc[1].num_samples = 5
#    enc[2].num_samples = 5  #//
#    enc[3].num_samples = 5
#    enc[4].num_samples = 5
#    enc6.num_samples = 5  #//
#    enc7.num_samples = 5
#    enc8.num_samples = 5
#    enc9.num_samples = 5  #//
#    enc10.num_samples = 5
#    enc11.num_samples = 5
#    enc12.num_samples = 5
#    enc13.num_samples = 5  #//
#    enc14.num_samples = 5
#    enc15.num_samples = 5
#    enc16.num_samples = 5  #//
#    enc17.num_samples = 5
#    enc18.num_samples = 5
#    enc19.num_samples = 5  #//
#    enc20.num_samples = 5


    graphsage = SupervisedGraphSage(7, enc[4]) #//
#    graphsage.cuda()
    #pdb.set_trace()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    #train = list(rand_indices[:])
    train = list(rand_indices[1500:])
#    print train
#    print "num_nodes1:", num_nodes
    #pdb.set_trace()

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:256]
        #print "batch_nodes:", batch_nodes
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.item()

    val_output = graphsage.forward(val) 
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)

    test_output = graphsage.forward(test) 
    print "Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    print feat_data[0]
    print labels[0]
    print adj_lists[0]
    return feat_data, labels, adj_lists

def run_cora():
    print "----------------cora----------------"
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
#    features.cuda()

    agg = []
    enc = []
    layer_num = 3
    layer_gap = (1433-128)/(layer_num)
    agg0 = MeanAggregator(features, cuda=True)
    enc0 = Encoder(features, 1433, 128+layer_gap*(layer_num-1), adj_lists, agg0, gcn=True, cuda=False)
    agg.append(agg0)
    enc.append(enc0)
    '''
    for i in range(1, layer_num):
       agg1 = MeanAggregator(lambda nodes: enc[i-1](nodes).t(), cuda=False)
       enc1 = Encoder(lambda nodes: enc[i-1](nodes).t(), 128 + layer_gap*(layer_num-i), 128 + layer_gap*(layer_num-1-i), \
               adj_lists, agg1, base_model=enc[i-1], gcn=True, cuda=False)
       enc1.num_samples = 5
       agg.append(agg1)
       enc.append(enc1)
       
    '''
    agg1 = MeanAggregator(lambda nodes: enc[0](nodes).t(), cuda=False)  #//
    enc1 = Encoder(lambda nodes: enc[0](nodes).t(), 128+layer_gap*(layer_num-1), 128+layer_gap*(layer_num-2), adj_lists, agg1,
            base_model=enc[0], gcn=True, cuda=False)  #//
    agg.append(agg1)
    enc.append(enc1)
    agg2 = MeanAggregator(lambda nodes: enc[1](nodes).t(), cuda=False)  #//
    enc2 = Encoder(lambda nodes: enc[1](nodes).t(), 128+layer_gap*(layer_num-2), 128+layer_gap*(layer_num-3), adj_lists, agg2,
            base_model=enc[1], gcn=True, cuda=False)  #//
    agg.append(agg2)
    enc.append(enc2)
    '''agg3 = MeanAggregator(lambda nodes: enc[2](nodes).t(), cuda=False)  #//
    enc3 = Encoder(lambda nodes: enc[2](nodes).t(), enc[2].embed_dim, 1208, adj_lists, agg3,
            base_model=enc[2], gcn=True, cuda=False)  #//
    agg.append(agg3)
    enc.append(enc3)
    agg4 = MeanAggregator(lambda nodes: enc[3](nodes).t(), cuda=False)  #//
    enc4 = Encoder(lambda nodes: enc[3](nodes).t(), enc[3].embed_dim, 1168, adj_lists, agg4,
            base_model=enc[3], gcn=True, cuda=False)  #//
    agg.append(agg4)
    enc.append(enc4)
    agg5 = MeanAggregator(lambda nodes: enc4(nodes).t(), cuda=False)  #//
    enc5 = Encoder(lambda nodes: enc4(nodes).t(), enc4.embed_dim, 1128, adj_lists, agg5,
            base_model=enc4, gcn=True, cuda=False)  #//
    
    
    agg6 = MeanAggregator(lambda nodes: enc5(nodes).t(), cuda=False)  #//
    enc6 = Encoder(lambda nodes: enc5(nodes).t(), enc5.embed_dim, 1088, adj_lists, agg6,
            base_model=enc5, gcn=True, cuda=False)  #//
    agg7 = MeanAggregator(lambda nodes: enc6(nodes).t(), cuda=False)  #//
    enc7 = Encoder(lambda nodes: enc6(nodes).t(), enc6.embed_dim, 1048, adj_lists, agg7,
            base_model=enc6, gcn=True, cuda=False)  #//
    agg8 = MeanAggregator(lambda nodes: enc7(nodes).t(), cuda=False)  #//
    enc8 = Encoder(lambda nodes: enc7(nodes).t(), enc7.embed_dim, 1008, adj_lists, agg8,
            base_model=enc7, gcn=True, cuda=False)  #//
    agg9 = MeanAggregator(lambda nodes: enc8(nodes).t(), cuda=False)  #//
    enc9 = Encoder(lambda nodes: enc8(nodes).t(), enc8.embed_dim, 968, adj_lists, agg9,
            base_model=enc8, gcn=True, cuda=False)  #//
    agg10 = MeanAggregator(lambda nodes: enc9(nodes).t(), cuda=False)  #//
    enc10 = Encoder(lambda nodes: enc9(nodes).t(), enc9.embed_dim, 928, adj_lists, agg10,
            base_model=enc9, gcn=True, cuda=False)  #//
    
    agg11 = MeanAggregator(lambda nodes: enc10(nodes).t(), cuda=False)  #//
    enc11 = Encoder(lambda nodes: enc10(nodes).t(), enc10.embed_dim, 888, adj_lists, agg11,
            base_model=enc10, gcn=True, cuda=False)  #//
    agg12 = MeanAggregator(lambda nodes: enc11(nodes).t(), cuda=False)
    enc12 = Encoder(lambda nodes: enc11(nodes).t(), enc11.embed_dim, 848, adj_lists, agg12, 
            base_model=enc11, gcn=True, cuda=False)
    agg13 = MeanAggregator(lambda nodes: enc12(nodes).t(), cuda=False)  #//
    enc13 = Encoder(lambda nodes: enc12(nodes).t(), enc12.embed_dim, 808, adj_lists, agg13,
            base_model=enc12, gcn=True, cuda=False)  #//
    agg14 = MeanAggregator(lambda nodes: enc13(nodes).t(), cuda=False)  #//
    enc14 = Encoder(lambda nodes: enc13(nodes).t(), enc13.embed_dim, 768, adj_lists, agg14,
            base_model=enc13, gcn=True, cuda=False)  #//
    agg15 = MeanAggregator(lambda nodes: enc14(nodes).t(), cuda=False)  #//
    enc15 = Encoder(lambda nodes: enc14(nodes).t(), enc14.embed_dim, 728, adj_lists, agg15,
            base_model=enc14, gcn=True, cuda=False)  #//
    agg16 = MeanAggregator(lambda nodes: enc15(nodes).t(), cuda=False)  #//
    enc16 = Encoder(lambda nodes: enc15(nodes).t(), enc15.embed_dim, 688, adj_lists, agg16,
            base_model=enc15, gcn=True, cuda=False)  #//
    agg17 = MeanAggregator(lambda nodes: enc16(nodes).t(), cuda=False)  #//
    enc17 = Encoder(lambda nodes: enc16(nodes).t(), enc16.embed_dim, 648, adj_lists, agg17,
            base_model=enc16, gcn=True, cuda=False)  #//
    agg18 = MeanAggregator(lambda nodes: enc17(nodes).t(), cuda=False)  #//
    enc18 = Encoder(lambda nodes: enc17(nodes).t(), enc17.embed_dim, 608, adj_lists, agg18,
            base_model=enc17, gcn=True, cuda=False)  #//
    agg19 = MeanAggregator(lambda nodes: enc18(nodes).t(), cuda=False)  #//
    enc19 = Encoder(lambda nodes: enc18(nodes).t(), enc18.embed_dim, 568, adj_lists, agg19,
            base_model=enc18, gcn=True, cuda=False)  #//
    agg20 = MeanAggregator(lambda nodes: enc19(nodes).t(), cuda=False)  #//
    enc20 = Encoder(lambda nodes: enc19(nodes).t(), enc19.embed_dim, 528, adj_lists, agg20,
            base_model=enc19, gcn=True, cuda=False)  #//
    agg21 = MeanAggregator(lambda nodes: enc20(nodes).t(), cuda=False)  #//
    enc21 = Encoder(lambda nodes: enc20(nodes).t(), enc20.embed_dim, 488, adj_lists, agg21,
            base_model=enc20, gcn=True, cuda=False)  #//
    agg22 = MeanAggregator(lambda nodes: enc21(nodes).t(), cuda=False)
    enc22 = Encoder(lambda nodes: enc21(nodes).t(), enc21.embed_dim, 448, adj_lists, agg22, 
            base_model=enc21, gcn=True, cuda=False)
    agg23 = MeanAggregator(lambda nodes: enc22(nodes).t(), cuda=False)  #//
    enc23 = Encoder(lambda nodes: enc22(nodes).t(), enc22.embed_dim, 408, adj_lists, agg23,
            base_model=enc22, gcn=True, cuda=False)  #//
    agg24 = MeanAggregator(lambda nodes: enc23(nodes).t(), cuda=False)  #//
    enc24 = Encoder(lambda nodes: enc23(nodes).t(), enc23.embed_dim, 368, adj_lists, agg24,
            base_model=enc23, gcn=True, cuda=False)  #//
    agg25 = MeanAggregator(lambda nodes: enc24(nodes).t(), cuda=False)  #//
    enc25 = Encoder(lambda nodes: enc24(nodes).t(), enc24.embed_dim, 328, adj_lists, agg25,
            base_model=enc24, gcn=True, cuda=False)  #//
    agg26 = MeanAggregator(lambda nodes: enc25(nodes).t(), cuda=False)  #//
    enc26 = Encoder(lambda nodes: enc25(nodes).t(), enc25.embed_dim, 288, adj_lists, agg26,
            base_model=enc25, gcn=True, cuda=False)  #//
    agg27 = MeanAggregator(lambda nodes: enc26(nodes).t(), cuda=False)  #//
    enc27 = Encoder(lambda nodes: enc26(nodes).t(), enc26.embed_dim, 248, adj_lists, agg27,
            base_model=enc26, gcn=True, cuda=False)  #//
    agg28 = MeanAggregator(lambda nodes: enc27(nodes).t(), cuda=False)  #//
    enc28 = Encoder(lambda nodes: enc27(nodes).t(), enc27.embed_dim, 208, adj_lists, agg28,
            base_model=enc27, gcn=True, cuda=False)  #//
    agg29 = MeanAggregator(lambda nodes: enc28(nodes).t(), cuda=False)  #//
    enc29 = Encoder(lambda nodes: enc28(nodes).t(), enc28.embed_dim, 168, adj_lists, agg29,
            base_model=enc28, gcn=True, cuda=False)  #//
    agg30 = MeanAggregator(lambda nodes: enc29(nodes).t(), cuda=False)  #//
    enc30 = Encoder(lambda nodes: enc29(nodes).t(), enc29.embed_dim, 128, adj_lists, agg30,
            base_model=enc29, gcn=True, cuda=False)  #//  '''

    #print "enc[4]:\n", enc[4]
    #print "enc[3]:\n", enc[3]
    print "enc[2]:\n", enc[2]
    print "enc[1]:\n", enc[1]
    print "enc[0]:\n", enc[0]
    enc[0].num_samples = 5
    enc[1].num_samples = 5
    enc[2].num_samples = 5  #//
    #enc[3].num_samples = 5
    #enc[4].num_samples = 5
#    enc6.num_samples = 5  #//
#    enc7.num_samples = 5
#    enc8.num_samples = 5
#    enc9.num_samples = 5  #//
#    enc10.num_samples = 5
#    enc11.num_samples = 5
#    enc12.num_samples = 5
#    enc13.num_samples = 5  #//
#    enc14.num_samples = 5
#    enc15.num_samples = 5
#    enc16.num_samples = 5  #//
#    enc17.num_samples = 5
#    enc18.num_samples = 5
#    enc19.num_samples = 5  #//
#    enc20.num_samples = 5


    graphsage = SupervisedGraphSage(7, enc[2]) #//
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    #train = list(rand_indices[1500:])
    train = list(rand_indices[1500:])
#    print train
#    print "num_nodes1:", num_nodes

    #pdb.set_trace()
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(400):
        batch_nodes = train[:256]
        #print "batch_nodes:", batch_nodes
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.item()

    val_output = graphsage.forward(val)
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Validation ErrorRate:", 1-accuracy_score(labels[val], val_output.data.numpy().argmax(axis=1))
    
    test_output = graphsage.forward(test) 
    print "Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
    print "Test ErrorRate:", 1-accuracy_score(labels[test], test_output.data.numpy().argmax(axis=1))
    print "Average batch time:", np.mean(times)
    
def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    print "----------------pubmed----------------"
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 360, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 240, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    agg3 = MeanAggregator(lambda nodes : enc2(nodes).t(), cuda=False)
    enc3 = Encoder(lambda nodes : enc2(nodes).t(), enc2.embed_dim, 128, adj_lists, agg3,
            base_model=enc2, gcn=True, cuda=False)

    enc1.num_samples = 10
    enc2.num_samples = 25
    enc3.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc3)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(400):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.item()

    val_output = graphsage.forward(val) 
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Validation ErrorRate:", 1-accuracy_score(labels[val], val_output.data.numpy().argmax(axis=1))
    
    test_output = graphsage.forward(test) 
    print "Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
    print "Test ErrorRate:", 1-accuracy_score(labels[test], test_output.data.numpy().argmax(axis=1))
    print "Average batch time:", np.mean(times)

if __name__ == "__main__":
    run_pubmed()
