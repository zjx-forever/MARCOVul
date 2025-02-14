import copy
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch_geometric.nn import GCNConv, GlobalAttention, RGATConv, RGCNConv, GATConv, SAGPooling, GIN


from einops import rearrange
import treelstm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.utils import calculate_evaluation_orders_cuda


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if isinstance(module, RGCNConv):
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if isinstance(module, RGCNConv) or isinstance(module, nn.Linear):
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


class MulModel_GCN_GCN_RGCN_LLM(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(MulModel_GCN_GCN_RGCN_LLM, self).__init__()
        self.g_type = list(config.g_type)

        self.use_text = config.use_text

        self._config_model = config.model

        if 'ast' in self.g_type:
            self.ast_readout = GlobalAttention(nn.Linear(256, 1))
            self.ast_GNN_1 = GCNConv(self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                                     1024)
            self.ast_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
            self.ast_GNN_2 = GCNConv(1024, 512)
            self.ast_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
            self.ast_GNN_3 = GCNConv(512, 256)

        if 'cfg' in self.g_type:
            self.cfg_readout = GlobalAttention(nn.Linear(256, 1))
            self.cfg_GNN_1 = GCNConv(self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                                     1024)
            self.cfg_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
            self.cfg_GNN_2 = GCNConv(1024, 512)
            self.cfg_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
            self.cfg_GNN_3 = GCNConv(512, 256)

        if 'pdg' in self.g_type:
            self.pdg_readout = GlobalAttention(nn.Linear(256, 1))
            self.pdg_GNN_1 = RGCNConv(self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                                      1024, num_relations=2)
            self.pdg_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
            self.pdg_GNN_2 = RGCNConv(1024, 512, num_relations=2)
            self.pdg_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
            self.pdg_GNN_3 = RGCNConv(512, 256, num_relations=2)

        
        MLP_input_size = 0
        for t in self.g_type:
            if t == 'ast':
                MLP_input_size += 256
            elif t == 'cfg':
                MLP_input_size += 256
            elif t == 'pdg':
                MLP_input_size += 256
        if self.use_text:
            MLP_input_size += 256
            self.LLM_Linear = nn.Linear(
                self._config_model.embedding[self._config_model.embedding.name_novar].embed_size, 256)

        
        MLP_layers = [
            nn.Linear(MLP_input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        ]

        self.MLPSequential = nn.Sequential(*MLP_layers)

        
        self.classifierLayer = nn.Linear(256, 2)

    def forward(self, data):
        ast_readout = None
        cfg_readout = None
        pdg_readout = None

        source_code = data[self.g_type[0]].func

        device = data[self.g_type[0]].x.device

        
        if 'ast' in self.g_type:
            ast_graph = data['ast']
            ast_node_inf = ast_graph.x  
            ast_edge_index = ast_graph.edge_index
            ast_batch = ast_graph.batch
            ast_edge_type = ast_graph.edge_type  
            
            ast_node_embedding = ast_node_inf

            
            
            ast_node_embedding_GNN_1 = F.leaky_relu(self.ast_GNN_1(ast_node_embedding, ast_edge_index))
            
            ast_node_embedding_GNN_1, ast_edge_index_GNN_1, ast_edge_type_GNN_1, ast_batch_GNN_1, _, _ = self.ast_pooling_GNN_1(
                ast_node_embedding_GNN_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
            ast_node_embedding_GNN_2 = F.leaky_relu(self.ast_GNN_2(ast_node_embedding_GNN_1, ast_edge_index_GNN_1))
            
            ast_node_embedding_GNN_2, ast_edge_index_GNN_2, ast_edge_type_GNN_2, ast_batch_GNN_2, _, _ = self.ast_pooling_GNN_2(
                ast_node_embedding_GNN_2, ast_edge_index_GNN_1, edge_attr=ast_edge_type_GNN_1, batch=ast_batch_GNN_1)
            ast_node_embedding_GNN_3 = F.leaky_relu(self.ast_GNN_3(ast_node_embedding_GNN_2, ast_edge_index_GNN_2))
            ast_readout = self.ast_readout(ast_node_embedding_GNN_3, ast_batch_GNN_2)

        
        if 'cfg' in self.g_type:
            cfg_graph = data['cfg']
            cfg_node_inf = cfg_graph.x
            cfg_edge_index = cfg_graph.edge_index
            cfg_batch = cfg_graph.batch
            cfg_edge_type = cfg_graph.edge_type  
            
            cfg_node_embedding = cfg_node_inf
            
            
            cfg_node_embedding_GNN_1 = F.leaky_relu(self.cfg_GNN_1(cfg_node_embedding, cfg_edge_index))
            
            cfg_node_embedding_GNN_1, cfg_edge_index_GNN_1, cfg_edge_type_GNN_1, cfg_batch_GNN_1, _, _ = self.cfg_pooling_GNN_1(
                cfg_node_embedding_GNN_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
            cfg_node_embedding_GNN_2 = F.leaky_relu(self.cfg_GNN_2(cfg_node_embedding_GNN_1, cfg_edge_index_GNN_1))
            
            cfg_node_embedding_GNN_2, cfg_edge_index_GNN_2, cfg_edge_type_GNN_2, cfg_batch_GNN_2, _, _ = self.cfg_pooling_GNN_2(
                cfg_node_embedding_GNN_2, cfg_edge_index_GNN_1, edge_attr=cfg_edge_type_GNN_1, batch=cfg_batch_GNN_1)
            cfg_node_embedding_GNN_3 = F.leaky_relu(self.cfg_GNN_3(cfg_node_embedding_GNN_2, cfg_edge_index_GNN_2))
            cfg_readout = self.cfg_readout(cfg_node_embedding_GNN_3, cfg_batch_GNN_2)

        
        if 'pdg' in self.g_type:
            pdg_graph = data['pdg']
            pdg_node_inf = pdg_graph.x
            pdg_edge_index = pdg_graph.edge_index
            pdg_batch = pdg_graph.batch
            pdg_edge_type = pdg_graph.edge_type  
            pdg_edge_type = self.reset_edge_type(pdg_edge_type)
            
            pdg_node_embedding = pdg_node_inf

            
            
            pdg_node_embedding_GNN_1 = F.leaky_relu(
                self.pdg_GNN_1(pdg_node_embedding, pdg_edge_index, edge_type=pdg_edge_type))
            
            pdg_node_embedding_GNN_1, pdg_edge_index_GNN_1, pdg_edge_type_GNN_1, pdg_batch_GNN_1, _, _ = self.pdg_pooling_GNN_1(
                pdg_node_embedding_GNN_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
            pdg_node_embedding_GNN_2 = F.leaky_relu(
                self.pdg_GNN_2(pdg_node_embedding_GNN_1, pdg_edge_index_GNN_1, edge_type=pdg_edge_type_GNN_1))
            
            pdg_node_embedding_GNN_2, pdg_edge_index_GNN_2, pdg_edge_type_GNN_2, pdg_batch_GNN_2, _, _ = self.pdg_pooling_GNN_2(
                pdg_node_embedding_GNN_2, pdg_edge_index_GNN_1, edge_attr=pdg_edge_type_GNN_1, batch=pdg_batch_GNN_1)
            pdg_node_embedding_GNN_3 = F.leaky_relu(
                self.pdg_GNN_3(pdg_node_embedding_GNN_2, pdg_edge_index_GNN_2, edge_type=pdg_edge_type_GNN_2))
            pdg_readout = self.pdg_readout(pdg_node_embedding_GNN_3, pdg_batch_GNN_2)

        
        readout_tensors = [t for t in [ast_readout, cfg_readout, pdg_readout] if t is not None]

        
        connectOut = torch.cat(readout_tensors, 1)

        
        if self.use_text:
            LLM_embeddings = self.LLM_Linear(source_code)
            connectOut = torch.cat((connectOut, LLM_embeddings), 1)

        
        MLPRes = self.MLPSequential(connectOut)  
        
        res = self.classifierLayer(MLPRes)
        return res

    
    def reset_edge_type(self, edge_type):
        unique_edge_types, inverse_indices = torch.unique(edge_type, return_inverse=True, sorted=True)
        new_edge_type = inverse_indices.to(torch.int64)
        return new_edge_type

    def cal_loss(self, pred, target, config, class_weights: list = [1.0, 1.0]):
        
        
        tensor_class_weights = torch.as_tensor(class_weights).to(pred.device)  
        loss_fn = torch.nn.CrossEntropyLoss(weight=tensor_class_weights)
        
        loss = loss_fn(pred, target)
        
        loss = loss + l1_regularization(self, config.L1_alpha) + l2_regularization(self, config.L2_alpha)
        return loss

class MulModel_GAT_GAT_RGAT_LLM(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(MulModel_GAT_GAT_RGAT_LLM, self).__init__()
        self.g_type = list(config.g_type)

        self.use_text = config.use_text

        self._config_model = config.model

        if 'ast' in self.g_type:
            self.ast_GNN_1 = GATConv(self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                                     1024, heads=1, concat=False)
            self.ast_pooling_GNN_1 = SAGPooling(1024, ratio=0.9)
            self.ast_GNN_2 = GATConv(1024, 512, heads=1, concat=False)
            self.ast_pooling_GNN_2 = SAGPooling(512, ratio=0.9)
            self.ast_GNN_3 = GATConv(512, 256, heads=1, concat=False)
        if 'cfg' in self.g_type:
            self.cfg_GCN_layers = 3
            self.cfg_GNN_1 = GATConv(self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                                     1024, heads=1, concat=False)
            self.cfg_pooling_GNN_1 = SAGPooling(1024, ratio=0.9)
            self.cfg_GNN_2 = GATConv(1024, 512, heads=1, concat=False)
            self.cfg_pooling_GNN_2 = SAGPooling(512, ratio=0.9)
            self.cfg_GNN_3 = GATConv(512, 256, heads=1, concat=False)
        if 'pdg' in self.g_type:
            self.pdg_GNN_1 = RGATConv(self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                                      1024, num_relations=2, heads=1, concat=False)
            self.pdg_pooling_GNN_1 = SAGPooling(1024, ratio=0.9)
            self.pdg_GNN_2 = RGATConv(1024, 512, num_relations=2, heads=1, concat=False)
            self.pdg_pooling_GNN_2 = SAGPooling(512, ratio=0.9)
            self.pdg_GNN_3 = RGATConv(512, 256, num_relations=2, heads=1, concat=False)

        

        self.readout = GlobalAttention(nn.Linear(256, 1))

        
        MLP_input_size = 0
        for t in self.g_type:
            if t == 'ast':
                MLP_input_size += 256
            elif t == 'cfg':
                MLP_input_size += 256
            elif t == 'pdg':
                MLP_input_size += 256
        if self.use_text:
            MLP_input_size = MLP_input_size + self._config_model.embedding[
                self._config_model.embedding.name_novar].embed_size

        
        MLP_layers = [
            nn.Linear(MLP_input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        ]

        self.MLPSequential = nn.Sequential(*MLP_layers)

        
        self.classifierLayer = nn.Linear(256, 2)

    def forward(self, data):
        ast_readout = None
        cfg_readout = None
        pdg_readout = None

        source_code = data[self.g_type[0]].func

        device = data[self.g_type[0]].x.device

        
        if 'ast' in self.g_type:
            ast_graph = data['ast']
            ast_node_inf = ast_graph.x  
            ast_edge_index = ast_graph.edge_index
            ast_batch = ast_graph.batch
            ast_edge_type = ast_graph.edge_type  
            
            ast_node_embedding = ast_node_inf
            
            
            ast_node_embedding_GNN_1 = F.leaky_relu(self.ast_GNN_1(ast_node_embedding, ast_edge_index))
            
            ast_node_embedding_GNN_1, ast_edge_index_GNN_1, ast_edge_type_GNN_1, ast_batch_GNN_1, _, _ = self.ast_pooling_GNN_1(
                ast_node_embedding_GNN_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
            ast_node_embedding_GNN_2 = F.leaky_relu(self.ast_GNN_2(ast_node_embedding_GNN_1, ast_edge_index_GNN_1))
            
            ast_node_embedding_GNN_2, ast_edge_index_GNN_2, ast_edge_type_GNN_2, ast_batch_GNN_2, _, _ = self.ast_pooling_GNN_2(
                ast_node_embedding_GNN_2, ast_edge_index_GNN_1, edge_attr=ast_edge_type_GNN_1, batch=ast_batch_GNN_1)
            ast_node_embedding_GNN_3 = F.leaky_relu(self.ast_GNN_3(ast_node_embedding_GNN_2, ast_edge_index_GNN_2))
            
            ast_readout_GNN = self.readout(ast_node_embedding_GNN_3, ast_batch_GNN_2)
            ast_readout = ast_readout_GNN

        
        if 'cfg' in self.g_type:
            cfg_graph = data['cfg']
            cfg_node_inf = cfg_graph.x
            cfg_edge_index = cfg_graph.edge_index
            cfg_batch = cfg_graph.batch
            cfg_edge_type = cfg_graph.edge_type  
            
            cfg_node_embedding = cfg_node_inf
            
            
            cfg_node_embedding_GNN_1 = F.leaky_relu(self.cfg_GNN_1(cfg_node_embedding, cfg_edge_index))
            
            cfg_node_embedding_GNN_1, cfg_edge_index_GNN_1, cfg_edge_type_GNN_1, cfg_batch_GNN_1, _, _ = self.cfg_pooling_GNN_1(
                cfg_node_embedding_GNN_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
            cfg_node_embedding_GNN_2 = F.leaky_relu(self.cfg_GNN_2(cfg_node_embedding_GNN_1, cfg_edge_index_GNN_1))
            
            cfg_node_embedding_GNN_2, cfg_edge_index_GNN_2, cfg_edge_type_GNN_2, cfg_batch_GNN_2, _, _ = self.cfg_pooling_GNN_2(
                cfg_node_embedding_GNN_2, cfg_edge_index_GNN_1, edge_attr=cfg_edge_type_GNN_1, batch=cfg_batch_GNN_1)
            cfg_node_embedding_GNN_3 = F.leaky_relu(self.cfg_GNN_3(cfg_node_embedding_GNN_2, cfg_edge_index_GNN_2))
            
            cfg_readout = self.readout(cfg_node_embedding_GNN_3, cfg_batch_GNN_2)

        
        if 'pdg' in self.g_type:
            pdg_graph = data['pdg']
            pdg_node_inf = pdg_graph.x
            pdg_edge_index = pdg_graph.edge_index
            pdg_batch = pdg_graph.batch
            pdg_edge_type = pdg_graph.edge_type  
            pdg_edge_type = self.reset_edge_type(pdg_edge_type, device)
            
            pdg_node_embedding = pdg_node_inf

            
            
            pdg_node_embedding_GNN_1 = F.leaky_relu(
                self.pdg_GNN_1(pdg_node_embedding, pdg_edge_index, edge_type=pdg_edge_type))
            
            pdg_node_embedding_GNN_1, pdg_edge_index_GNN_1, pdg_edge_type_GNN_1, pdg_batch_GNN_1, _, _ = self.pdg_pooling_GNN_1(
                pdg_node_embedding_GNN_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
            pdg_node_embedding_GNN_2 = F.leaky_relu(
                self.pdg_GNN_2(pdg_node_embedding_GNN_1, pdg_edge_index_GNN_1, edge_type=pdg_edge_type_GNN_1))
            
            pdg_node_embedding_GNN_2, pdg_edge_index_GNN_2, pdg_edge_type_GNN_2, pdg_batch_GNN_2, _, _ = self.pdg_pooling_GNN_2(
                pdg_node_embedding_GNN_2, pdg_edge_index_GNN_1, edge_attr=pdg_edge_type_GNN_1, batch=pdg_batch_GNN_1)
            pdg_node_embedding_GNN_3 = F.leaky_relu(
                self.pdg_GNN_3(pdg_node_embedding_GNN_2, pdg_edge_index_GNN_2, edge_type=pdg_edge_type_GNN_2))
            
            pdg_readout_GNN = self.readout(pdg_node_embedding_GNN_3, pdg_batch_GNN_2)
            pdg_readout = pdg_readout_GNN

        
        readout_tensors = [t for t in [ast_readout, cfg_readout, pdg_readout] if t is not None]

        
        connectOut = torch.cat(readout_tensors, 1)

        
        
        if self.use_text:
            word2vec_embeddings = source_code
            connectOut = torch.cat((connectOut, word2vec_embeddings), 1)

        
        MLPRes = self.MLPSequential(connectOut)  
        
        res = self.classifierLayer(MLPRes)
        return res

    
    def reset_edge_type(self, edge_type, device):
        unique_edge_types, inverse_indices = torch.unique(edge_type, return_inverse=True, sorted=True)
        new_edge_type = inverse_indices.to(torch.int64)
        return new_edge_type

    
    def sourse_code_embedding(self, func_list):
        
        word2vec_embeddings = self._embedding(func_list)
        return word2vec_embeddings

    def cal_loss(self, pred, target, config, class_weights: list = [1.0, 1.0]):
        
        
        tensor_class_weights = torch.as_tensor(class_weights).to(pred.device)  
        loss_fn = torch.nn.CrossEntropyLoss(weight=tensor_class_weights)
        
        loss = loss_fn(pred, target)
        
        loss = loss + l1_regularization(self, config.L1_alpha) + l2_regularization(self, config.L2_alpha)
        return loss

class Single_Text_LLM(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(Single_Text_LLM, self).__init__()
        self.g_type = list(config.g_type)[0]

        self.LLM_Linear = nn.Linear(config.model.embedding[config.model.embedding.name_novar].embed_size,
                                    256)

        
        MLP_input_size = 256

        
        MLP_layers = [
            nn.Linear(MLP_input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        ]

        self.MLPSequential = nn.Sequential(*MLP_layers)

        
        self.classifierLayer = nn.Linear(256, 2)

    def forward(self, data):
        source_code = data[self.g_type].func

        LLM_embeddings = self.LLM_Linear(source_code)

        
        MLPRes = self.MLPSequential(LLM_embeddings)  
        
        res = self.classifierLayer(MLPRes)
        return res

    def cal_loss(self, pred, target, config, class_weights: list = [1.0, 1.0]):
        
        
        tensor_class_weights = torch.as_tensor(class_weights).to(pred.device)  
        loss_fn = torch.nn.CrossEntropyLoss(weight=tensor_class_weights)
        
        loss = loss_fn(pred, target)
        
        loss = loss + l1_regularization(self, config.L1_alpha) + l2_regularization(self, config.L2_alpha)
        return loss

class MulModel_Four_modules_LLM(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(MulModel_Four_modules_LLM, self).__init__()
        self.g_type = list(config.g_type)

        self.use_text = config.use_text

        self._config_model = config.model

        self._config_pre_train_structure = config.pre_train_structure

        if 'ast' in self.g_type:
            self.ast_readout = GlobalAttention(nn.Linear(256, 1))
            
            self.ast_TreeLSTM_1 = treelstm.TreeLSTM(
                self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size, 1024)
            self.ast_TreeLSTM_2 = treelstm.TreeLSTM(1024, 512)
            self.ast_TreeLSTM_3 = treelstm.TreeLSTM(512, 256)

        if 'cfg' in self.g_type:
            self.cfg_readout = GlobalAttention(nn.Linear(256, 1))
            
            self.cfg_GNN_1 = GATConv(self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                                     1024, heads=3, concat=False)
            self.cfg_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
            self.cfg_GNN_2 = GATConv(1024, 512, heads=3, concat=False)
            self.cfg_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
            self.cfg_GNN_3 = GATConv(512, 256, heads=3, concat=False)

        if 'pdg' in self.g_type:
            self.pdg_readout = GlobalAttention(nn.Linear(256, 1))
            
            self.pdg_GIN = GIN(
                in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                hidden_channels=1024, out_channels=256, num_layers=3,
                dropout=0, act=F.leaky_relu)  

        
        MLP_input_size = 0
        for t in self.g_type:
            if t == 'ast':
                MLP_input_size += 256
            elif t == 'cfg':
                MLP_input_size += 256
            elif t == 'pdg':
                MLP_input_size += 256
        if self.use_text:
            MLP_input_size += 256
            self.LLM_Linear = nn.Linear(self._config_model.embedding[self._config_model.embedding.name_novar].embed_size, 256)

        
        if self._config_pre_train_structure.used:
            
            ast_cfg_pdg_MLP = [
                nn.Linear(256, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
                nn.Dropout(0.1),

                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ]
            self.ast_MLP = nn.Sequential(*copy.deepcopy(ast_cfg_pdg_MLP))
            self.cfg_MLP = nn.Sequential(*copy.deepcopy(ast_cfg_pdg_MLP))
            self.pdg_MLP = nn.Sequential(*copy.deepcopy(ast_cfg_pdg_MLP))

            
            self.all_self_attention = SelfAttention_MulHead(256, num_heads=2)

            data_type = self.g_type
            if self.use_text:
                data_type.append('text')
            self.att_n_m = Attention_N_module(data_type, 256, 1024, concat=True)

            MLP_layers = [
                
                nn.Linear(4096, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
                nn.Dropout(0.1),

                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            ]

            self.MLPSequential = nn.Sequential(*MLP_layers)

            
            self.classifierLayer = nn.Linear(256, 2)

        else:
            MLP_layers = [
                nn.Linear(MLP_input_size, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
                nn.Dropout(0.1),

                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ]

            self.MLPSequential = nn.Sequential(*MLP_layers)

            
            self.classifierLayer = nn.Linear(256, 2)

    def forward(self, data):
        ast_readout = None
        cfg_readout = None
        pdg_readout = None

        source_code = data[self.g_type[0]].func

        device = data[self.g_type[0]].x.device

        def ast_module():
            ast_graph = data['ast']
            ast_node_inf = ast_graph.x  
            ast_edge_index = ast_graph.edge_index
            ast_batch = ast_graph.batch
            ast_edge_type = ast_graph.edge_type  
            
            ast_node_embedding = ast_node_inf
            
            
            node_order, edge_order = calculate_evaluation_orders_cuda(ast_edge_index.t(), len(ast_node_inf), device)
            ast_TreeLSTM_out_h_1, _ = self.ast_TreeLSTM_1(ast_node_embedding.float(), node_order, ast_edge_index.t(),
                                                          edge_order)
            ast_TreeLSTM_out_h_2, _ = self.ast_TreeLSTM_2(ast_TreeLSTM_out_h_1, node_order, ast_edge_index.t(),
                                                          edge_order)
            ast_TreeLSTM_out_h_3, _ = self.ast_TreeLSTM_3(ast_TreeLSTM_out_h_2, node_order, ast_edge_index.t(),
                                                          edge_order)
            ast_readout = self.ast_readout(ast_TreeLSTM_out_h_3, ast_batch)
            return ast_readout

        def cfg_module():
            cfg_graph = data['cfg']
            cfg_node_inf = cfg_graph.x
            cfg_edge_index = cfg_graph.edge_index
            cfg_batch = cfg_graph.batch
            cfg_edge_type = cfg_graph.edge_type  
            
            cfg_node_embedding = cfg_node_inf
            
            
            cfg_node_embedding_GNN_1 = F.leaky_relu(self.cfg_GNN_1(cfg_node_embedding, cfg_edge_index))
            
            cfg_node_embedding_GNN_1, cfg_edge_index_GNN_1, cfg_edge_type_GNN_1, cfg_batch_GNN_1, _, _ = self.cfg_pooling_GNN_1(
                cfg_node_embedding_GNN_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
            cfg_node_embedding_GNN_2 = F.leaky_relu(self.cfg_GNN_2(cfg_node_embedding_GNN_1, cfg_edge_index_GNN_1))
            
            cfg_node_embedding_GNN_2, cfg_edge_index_GNN_2, cfg_edge_type_GNN_2, cfg_batch_GNN_2, _, _ = self.cfg_pooling_GNN_2(
                cfg_node_embedding_GNN_2, cfg_edge_index_GNN_1, edge_attr=cfg_edge_type_GNN_1,
                batch=cfg_batch_GNN_1)
            cfg_node_embedding_GNN_3 = F.leaky_relu(self.cfg_GNN_3(cfg_node_embedding_GNN_2, cfg_edge_index_GNN_2))
            cfg_readout = self.cfg_readout(cfg_node_embedding_GNN_3, cfg_batch_GNN_2)
            return cfg_readout

        def pdg_module():
            pdg_graph = data['pdg']
            pdg_node_inf = pdg_graph.x
            pdg_edge_index = pdg_graph.edge_index
            pdg_batch = pdg_graph.batch
            pdg_edge_type = pdg_graph.edge_type  
            pdg_edge_type = self.reset_edge_type(pdg_edge_type, device)
            
            pdg_node_embedding = pdg_node_inf

            
            
            pdg_node_embedding_GIN = self.pdg_GIN(pdg_node_embedding, pdg_edge_index, batch=pdg_batch,
                                                  batch_size=pdg_batch.max().item() + 1)
            pdg_readout = self.pdg_readout(pdg_node_embedding_GIN, pdg_batch)
            return pdg_readout

        
        if 'ast' in self.g_type:
            if self._config_pre_train_structure.used:
                
                ast_readout = ast_module()
                ast_readout = self.ast_MLP(ast_readout)
            else:
                ast_readout = ast_module()

        
        if 'cfg' in self.g_type:
            if self._config_pre_train_structure.used:
                
                cfg_readout = cfg_module()
                cfg_readout = self.cfg_MLP(cfg_readout)
            else:
                cfg_readout = cfg_module()

        
        if 'pdg' in self.g_type:
            if self._config_pre_train_structure.used:
                
                pdg_readout = pdg_module()
                pdg_readout = self.pdg_MLP(pdg_readout)
            else:
                pdg_readout = pdg_module()

        
        readout_tensors = [t for t in [ast_readout, cfg_readout, pdg_readout] if t is not None]

        
        if self.use_text:
            LLM_embeddings = self.LLM_Linear(source_code)
            readout_tensors.append(LLM_embeddings)

        
        readout_tensors = torch.stack(readout_tensors, dim=1)
        self_att_out = self.all_self_attention(readout_tensors)
        self_att_out_list = torch.unbind(self_att_out, dim=1)

        
        att_merge_out = self.att_n_m(self_att_out_list)

        connectOut = att_merge_out

        
        MLPRes = self.MLPSequential(connectOut)  
        
        res = self.classifierLayer(MLPRes)
        return res

    
    def reset_edge_type(self, edge_type, device):
        unique_edge_types, inverse_indices = torch.unique(edge_type, return_inverse=True, sorted=True)
        new_edge_type = inverse_indices.to(torch.int64)
        return new_edge_type

    def cal_loss(self, pred, target, config, class_weights: list = [1.0, 1.0]):
        
        
        tensor_class_weights = torch.as_tensor(class_weights).to(pred.device)  
        loss_fn = torch.nn.CrossEntropyLoss(weight=tensor_class_weights)
        
        loss = loss_fn(pred, target)
        
        loss = loss + l1_regularization(self, config.L1_alpha) + l2_regularization(self, config.L2_alpha)
        return loss

class SelfAttention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads = 1):
        super().__init__()
        self.heads = num_heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, hidden_dim * num_heads * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SelfAttention_MulHead(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _, heads = *x.shape, self.num_heads
        qkv = self.to_qkv(x).view(batch_size, seq_len, 3, heads, self.head_dim)

        
        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]

        
        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)  

        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.head_dim ** -0.5
        attn = dots.softmax(dim=-1)

        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous()

        
        out = out.view(batch_size, seq_len, self.embed_dim)
        out = self.to_out(out)

        return out

class Attention_N_module(nn.Module):
    def __init__(self, data_type, ori_dim, map_dim, concat=False):
        super().__init__()
        self.num_modules = len(data_type)
        self.data_type = data_type
        self.map_dim = map_dim
        self.concat = concat
        
        ast_cfg_pdg_text_F = [
            nn.Linear(ori_dim, map_dim),
            nn.BatchNorm1d(map_dim),
            nn.LeakyReLU()
        ]
        ast_cfg_pdg_text_att = [
            nn.Linear(ori_dim*self.num_modules, map_dim),
            nn.BatchNorm1d(map_dim),
            
        ]

        self.attention_F = nn.ModuleList([nn.Sequential(*copy.deepcopy(ast_cfg_pdg_text_F)) for _ in range(self.num_modules)])
        self.attention = nn.ModuleList([nn.Sequential(*copy.deepcopy(ast_cfg_pdg_text_att)) for _ in range(self.num_modules)])

    def forward(self, X_list):

        all_embedding = torch.cat(X_list, -1)

        attention_F = [self.attention_F[i](X_list[i]) for i in range(self.num_modules)]

        attention = [self.attention[i](all_embedding) for i in range(self.num_modules)]

        if self.concat:
            all_embedding_F = torch.cat(attention_F, -1)
            att_deal = F.sigmoid(torch.cat(attention, dim=-1))
            all_out = all_embedding_F * att_deal
        else:
            all_embedding_F = torch.cat(attention_F, -1)
            att_deal = F.softmax(torch.cat(attention, dim=-1), dim=-1)
            all_out = (all_embedding_F * att_deal).chunk(self.num_modules, dim = -1)
            all_out = torch.sum(torch.stack(all_out, dim=0), dim=0)

            
            
            
        return all_out
