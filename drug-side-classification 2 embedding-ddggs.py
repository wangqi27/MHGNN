#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.sparse as sp
import numpy as np
import torch
import pandas as pd
import pickle
import dgl
import dgl.nn as dglnn
import numpy as np
from dgl.nn.pytorch import GATConv
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook
import pickle as pkl
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ## Data

# ### Load graph

# In[3]:


g = dgl.load_graphs('/Users/tlx/PythonFile/程序-副作用预测/fin_data/graph_data/graph_cs_1')[0][0]
g


# ### Load CSV data

# In[4]:


net_comb_side = pd.read_csv('/Users/tlx/PythonFile/程序-副作用预测/fin_data/fin_data/net_comb_side_1.csv')
net_comb_side.head()


# In[5]:


all_sides = list(set(net_comb_side['No_side']))
len(all_sides)


# In[6]:


min(net_comb_side.No_comb), max(net_comb_side.No_comb), min(net_comb_side.No_side), max(net_comb_side.No_side)


# In[7]:


node_mapping = pd.read_csv('/Users/tlx/PythonFile/程序-副作用预测/fin_data/fin_data/all_comb_no.csv')
node_mapping.head()


# ### Load embeddings

# In[8]:


with open('/Users/tlx/PythonFile/程序-副作用预测/drug-side-effect(1)/pretrained_graph/embeddings_ddggs_1.pkl', 'rb') as fin:
    embeddings = pkl.load(fin)
    
embeddings = {k: v.detach().numpy() for k, v in embeddings.items()}
embeddings


# In[9]:


embeddings['drug'].sum(axis=1).max(), embeddings['drug'].sum(axis=1).min()


# ### comb-nodes mapping

# In[10]:


# data_mapping = {row['No_comb']: f"{row['S1_No']},{row['S2_No']}" for idx, row in tqdm_notebook(net_comb_side.iterrows(), total=len(net_comb_side))}


# In[11]:


# with open('node_mapping.pkl', 'wb') as fout:
#     pkl.dump(data_mapping, fout)


# In[12]:


feature_dim = 128
h_dim = 128
comb_features = np.random.randn(207689, feature_dim)
side_features = np.random.randn(9968, feature_dim)
comb_features.shape, side_features.shape


# In[13]:


with open('/Users/tlx/PythonFile/程序-副作用预测/drug-side-effect/node_mapping.pkl', 'rb') as fin:
    data_mapping = pkl.load(fin)

side_features = embeddings['side'][:-1]

for k, v in tqdm_notebook(data_mapping.items(), total=len(data_mapping)):
    v0, v1 = [int(float(i)) for i in v.split(',')]
    comb_features[int(k)] = embeddings['drug'][v0] + embeddings['drug'][v1]


# In[14]:


g.nodes['comb'].data['h'] = torch.FloatTensor(comb_features)
g.nodes['side'].data['h'] = torch.FloatTensor(side_features)


# In[15]:


# train_sample = torch.zeros(len(comb_features), dtype=torch.bool).bernoulli(0.8)
# g.nodes['comb'].data['train_mask'] = train_sample
# g.nodes['comb'].data['test_mask'] = ~train_sample


# ## Training and evaluation

# ### Build model

# In[16]:


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names, dropout=0.):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.conv1 = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, hidden_feat, norm='right', weight=True)
                for rel in rel_names
            })
        self.conv2 = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(hidden_feat, out_feat, norm='right', weight=True)
                for rel in rel_names
            })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = {k: self.dropout1(v) for k, v in x.items()}
        x = self.conv2(blocks[1], x)
        x = {k: self.dropout2(v) for k, v in x.items()}
        return x
       
        
class ScorePredictor(nn.Module):
    
    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('h', 'h', 'score'), 
#                     self.apply_edges, 
                    etype=etype
                )
            return edge_subgraph.edata['score']

        
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, etypes, dropout=0.):
        super().__init__()
#         self.embed = torch.nn.Parameter(torch.Tensor(source_dim, embed_dim))
        self.rgcn = StochasticTwoLayerRGCN(
            in_features, hidden_features, out_features, etypes, dropout=dropout)
        self.pred = ScorePredictor()

    def forward(self, graph, blocks, x):
        x = self.rgcn(blocks, x)
        score = self.pred(graph, x)
        score = {etype: torch.sigmoid(value) for etype, value in score.items()}
        return score


# In[29]:


def compute_loss(pos_score, neg_score, margin=0):
    # 损失
    loss = []
    for etype, d in pos_score.items():
        n_edges_pos = pos_score[etype].shape[0]
        n_edges_neg = neg_score[etype].shape[0]
        if n_edges_pos == 0 or n_edges_neg == 0: continue
        loss.append((1 - (pos_score[etype].unsqueeze(1) + margin) + (neg_score[etype].view(n_edges_neg, -1) - margin)).clamp(min=0).mean())
#         loss.append((torch.sigmoid(- pos_score[etype].unsqueeze(1)) + torch.sigmoid(neg_score[etype].view(n_edges_neg, -1))).mean())
    
    if len(loss) == 0:
        raise ValueError('error value')
        
    loss = torch.hstack(loss).sum()
    return loss


def auprc_auroc(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    auroc, auprc = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
#     y, xx, _ = metrics.precision_recall_curve(y, pred)
#     auprc = metrics.auc(xx, y)

    return auprc, auroc


from sklearn import metrics
def compute_auprc_auroc(pos_score, neg_score):
    pos_label = torch.ones(pos_score.shape[0])
    neg_label = torch.zeros(neg_score.shape[0])
    
    pred = torch.cat([pos_score, neg_score])
    label = torch.cat([pos_label, neg_label])
    
    metric = auprc_auroc(label, pred)
    auprc, auroc = metric

    return auprc, auroc


from operator import itemgetter
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


# ### Build dataloader

# In[18]:


batch_size = 2048 #1024
test_batch_size = 4096


# In[19]:


eid_dict = {etype: g.edges(etype=etype, form='eid') for etype in ['effect']}

train_mask = torch.zeros(len(eid_dict['effect']), dtype=torch.bool).bernoulli(0.8)
test_mask = ~train_mask

train_eid_dict = {'effect': eid_dict['effect'][train_mask]}

test_eid_dict = {'effect': eid_dict['effect'][test_mask]}

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)


# In[20]:


# for eid in test_eid_dict['effect']:
#     src, dst = g.find_edges(eid, etype='effect')
#     if dst in all_sides:
#         print(eid, src, dst)
#     break


# In[21]:


# 用于存放每个side相关的边，key为side id，value为边集合
test_eid_by_side = {}
for s in tqdm_notebook(all_sides):
    s = int(s)
    part_eids = dgl.in_subgraph(g, {'side':[s]}).edges['effect'].data['_ID']
    part_eids = set(part_eids.tolist()).intersection(test_eid_dict['effect'].tolist())
    test_eid_by_side[s] = list(part_eids)


# In[22]:


test_dataloader_by_side = {}
for s, eids in tqdm_notebook(test_eid_by_side.items()):
    test_dataloader = dgl.dataloading.DistEdgeDataLoader(
        g, {'effect': eids}, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
#         num_workers=args.num_workers
    )
    
    test_dataloader_by_side[s] = test_dataloader


# * Remove isolated nodes

# In[23]:


need_remove = list(set(range(9968)).difference(list(map(int, all_sides))))
need_remove = np.random.choice(need_remove, 9968 - 963 * 1)
g.remove_nodes(torch.tensor(need_remove), ntype='side')
g


# In[24]:


# new_g = dgl.metapath_reachable_graph(g, ['similar'])
# ('drug', 'effect', 'side'): 169906, ('drug', 'has', 'gene'): 18632, ('drug', 'similar', 'drug'): 52557

# test_dataloader = dgl.dataloading.DistEdgeDataLoader(
#     g, test_eid_dict, sampler,
#     negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
#     batch_size=test_batch_size,
#     shuffle=False,
#     drop_last=False,
# #     num_workers=args.num_workers
# )


# ### Training

# In[30]:


from sklearn.model_selection import KFold

dropout=0.4

ids = train_eid_dict['effect'].tolist()
kf = KFold(n_splits=5)
margin = 0.15
models = []

for fold, (train_ids, val_ids) in enumerate(kf.split(ids)):
    print(f'######## fold {fold + 1}/5 ########')
    
    ### Define dataset ###
    train_eid_dict = {'effect': torch.tensor(train_ids)}
    val_eid_dict = {'effect': torch.tensor(val_ids)}
    
    dataloader = dgl.dataloading.DistEdgeDataLoader(
    g, train_eid_dict, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
#     num_workers=args.num_workers
    )
    
    val_dataloader = dgl.dataloading.DistEdgeDataLoader(
    g, val_eid_dict, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        batch_size=test_batch_size * 4,
        shuffle=True,
        drop_last=False,
#     num_workers=args.num_workers
    )
    
    ### Define model and optimizer ###
    model = Model(feature_dim, h_dim, h_dim, g.etypes, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr = 1e-3
    #### Start training ###
    step = 0
    for i in range(1):
        print(f'#### epoch {i + 1}/1 ####')
        bar = tqdm_notebook(dataloader, total=len(train_eid_dict['effect']) // batch_size)

        for input_nodes, positive_graph, negative_graph, blocks in bar:
            step += 1
        #     blocks = [b.to(torch.device('cuda')) for b in blocks]
        #     positive_graph = positive_graph.to(torch.device('cuda'))
        #     negative_graph = negative_graph.to(torch.device('cuda'))
            input_features = blocks[0].srcdata['h']
            pos_score = model(positive_graph, blocks, input_features)
            neg_score = model(negative_graph, blocks, input_features)

            loss = compute_loss(pos_score, neg_score, margin=margin)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:   
                for params in opt.param_groups:   
                    params['lr'] *= 0.95   
                    lr = params['lr']
            loss_val = loss.detach().item()
            
            ### Validation ###
            if step % 10 == 0:
                input_nodes, positive_graph, negative_graph, blocks = next(iter(val_dataloader))
                input_features = blocks[0].srcdata['h']
                pos_score = model(positive_graph, blocks, input_features)
                neg_score = model(negative_graph, blocks, input_features)
                pos_score_ = pos_score[('comb','effect','side')].detach().numpy().reshape(-1)
                neg_score_ = neg_score[('comb','effect','side')].detach().numpy().reshape(-1)

                eid = 0
                predicted = []
                actual = []
                for x in pos_score_:
                    predicted.append((x, eid))
                    actual.append(eid)
                    eid += 1

                for x in neg_score_:
                    predicted.append((x, eid))
                    eid += 1

                labels_all = np.hstack([np.ones(len(pos_score_)), np.zeros(len(neg_score_))])
                predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

                auprc, auroc = compute_auprc_auroc(pos_score[('comb','effect','side')], neg_score[('comb','effect','side')])
                ap = apk(actual, predicted, k=50)
                
                print(pos_score_, np.min(pos_score_), np.max(pos_score_))
                print(neg_score_, np.min(neg_score_), np.max(neg_score_))
                print(f'train loss:{round(loss_val, 4)}, auprc:{round(np.mean(auprc), 4)}, auroc:{round(np.mean(auroc), 4)}, ap@50:{round(np.mean(ap), 4)}')
                print('=' * 80)
            bar.set_description(f'loss:{round(loss_val, 4)}, lr:{round(lr, 4)}')
    models.append(model)


# In[32]:


torch.save(model, '/Users/tlx/PythonFile/程序-副作用预测/drug-side-effect(1)/cls_result/cls_model_ddggs_1.pt')


# ### Evaluation

# In[33]:


def cv_predict(models, pos_graph, neg_graph, blocks, input_features):
    folds = len(models)
    assert folds >= 1, ValueError('models cannot be empty')
    pos_score = None
    neg_score = None
    for m in models:
        pos = m(pos_graph, blocks, input_features)
        neg = m(neg_graph, blocks, input_features)
        
        pos = pos[('comb','effect','side')].detach().numpy().reshape(-1)
        neg = neg[('comb','effect','side')].detach().numpy().reshape(-1)
        
        if pos_score is None:
            pos_score = pos
            neg_score = neg
        else:
            pos_score += pos
            neg_score += neg
    return pos_score / folds, neg_score / folds


# #### For type

# In[34]:


#### Start test ###
tp_auprc = []
tp_auroc = []
tp_apk = []
for tp, loader in test_dataloader_by_side.items():
    print(f'current: {tp}')
    
    bar = tqdm_notebook(loader)
    auprcs = []
    aurocs = []
    aps = []
    pos_scores = []
    neg_scores = []
    actuals = []
    predictions = []
    eid = 0
    for input_nodes, positive_graph, negative_graph, blocks in bar:
        step += 1
        input_features = blocks[0].srcdata['h']
#         pos_score = model(positive_graph, blocks, input_features)
#         neg_score = model(negative_graph, blocks, input_features)
#         pos_score_ = pos_score[('comb','effect','side')].detach().numpy().reshape(-1)
#         neg_score_ = neg_score[('comb','effect','side')].detach().numpy().reshape(-1)
        pos_score, neg_score = cv_predict(models, positive_graph, negative_graph, blocks, input_features)
            
        predict = []
        actual = []
        for x in pos_score:
            predict.append((x, eid))
            actual.append(eid)
            eid += 1

        for x in neg_score:
            predict.append((x, eid))
            eid += 1
        
        pos_scores.extend(pos_score.tolist())
        neg_scores.extend(neg_score.tolist())
        predictions.extend(predict)
        actuals.extend(actual)

        predict = list(zip(*sorted(predict, reverse=True, key=itemgetter(0))))[1]

        auprc, auroc = compute_auprc_auroc(torch.tensor(pos_score), torch.tensor(neg_score))
        ap50 = apk(actual, predict, k=50)

#         auprcs.append(auprc)
#         aurocs.append(auroc)
#         aps.append(ap50)
        bar.set_description(f'auprc:{round(np.mean(auprc), 4)}, auroc:{round(np.mean(auroc), 4)}, ap@50:{round(np.mean(ap50), 4)}')


    predictions = list(zip(*sorted(predictions, reverse=True, key=itemgetter(0))))[1]
    auprc, auroc = compute_auprc_auroc(torch.tensor(pos_scores), torch.tensor(neg_scores))
    ap50 = apk(actuals, predictions, k=50)
    tp_auprc.append(auprc)
    tp_auroc.append(auroc)
    tp_apk.append(ap50)
    print(f'For type {tp}, metrics:: auprc:{round(np.mean(auprc), 4)}, auroc:{round(np.mean(auroc), 4)}, ap@50:{round(np.mean(ap50), 4)}')


# In[ ]:





# In[ ]:





# #### For batch

# In[ ]:


#### Start test ###
auprcs = []
aurocs = []
aps = []

bar = tqdm_notebook(test_dataloader, total=len(test_eid_dict['effect']) // batch_size)

actuals = []
predictions = []
eid = 0
for input_nodes, positive_graph, negative_graph, blocks in bar:
    step += 1
    input_features = blocks[0].srcdata['h']
    pos_score = model(positive_graph, blocks, input_features)
    neg_score = model(negative_graph, blocks, input_features)

    pos_score_ = pos_score[('comb','effect','side')].detach().numpy().reshape(-1)
    neg_score_ = neg_score[('comb','effect','side')].detach().numpy().reshape(-1)

    
    predict = []
    actual = []
    for x in pos_score_:
        predict.append((x, eid))
        actual.append(eid)
        eid += 1

    for x in neg_score_:
        predict.append((x, eid))
        eid += 1
    
    predictions.extend(predict)
    actuals.extend(actual)
    
    predict = list(zip(*sorted(predict, reverse=True, key=itemgetter(0))))[1]

    auprc, auroc = compute_auprc_auroc(pos_score[('comb','effect','side')], neg_score[('comb','effect','side')])
    ap50 = apk(actual, predict, k=50)

    auprcs.append(auprc)
    aurocs.append(auroc)
    aps.append(ap50)
    bar.set_description(f'auprc:{round(np.mean(auprcs), 4)}, auroc:{round(np.mean(aurocs), 4)}, ap@50:{round(np.mean(aps), 4)}')
    

predictions = list(zip(*sorted(predictions, reverse=True, key=itemgetter(0))))[1]
auprc, auroc = compute_auprc_auroc(pos_score[('comb','effect','side')], neg_score[('comb','effect','side')])
ap50 = apk(actual, predict, k=50)
print(auprc, auroc, ap50)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# -----

# In[ ]:


# ddgg_1 loss:0.0979, auprc:0.8757, auroc:0.9728, ap:0.7924


# In[26]:


# edges = []
# src_nodes = []
# dst_nodes = []
# ps = []
# for i, p in tqdm_notebook(zip(positive_graph.edges['effect'].data['_ID'], positive_graph.edges['effect'].data['p'])):
#     edges.append(i.item())
#     src_node, dst_node = g_.find_edges(i.item(), etype='effect')
# #     node_pairs.append([src_node.item(), dst_node.item()])
#     src_nodes.append(src_node.item())
#     dst_nodes.append(dst_node.item())
#     ps.append(p.item())
# #     g_.edges[src_node.item(), dst_node.item()].data['p'] = p

# g_.edges[src_nodes, dst_nodes].data['p'] = torch.tensor(ps)


# In[27]:


# all_sides = set(net_comb_side['No_side'])

# edges = []
# src_nodes = []
# dst_nodes = []
# side_scores = {}
# for i, p in tqdm_notebook(zip(positive_graph.edges['effect'].data['_ID'], positive_graph.edges['effect'].data['p'])):
#     edges.append(i.item())
#     src_node, dst_node = g_.find_edges(i.item(), etype='effect')
#     src_node = src_node.item()
#     dst_node = dst_node.item()
#     if dst_node  in all_sides:
#         side_scores.setdefault(dst_node, [])
#         side_scores[dst_node].append(p)



# In[ ]:





# In[ ]:




