import copy
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch_geometric.nn import GCNConv, GlobalAttention, RGCNConv, GATConv, SAGPooling, ResGatedGraphConv, GIN, \
    FiLMConv, GMMConv, TransformerConv, TAGConv, SAGEConv

from einops import rearrange
import treelstm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.utils import calculate_evaluation_orders_cuda

modelTuple = (
treelstm.TreeLSTM, GCNConv, RGCNConv, GATConv, ResGatedGraphConv, GIN, FiLMConv, GMMConv, TransformerConv, TAGConv,
SAGEConv)


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
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


class MulModel_Single_Test_LLM(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(MulModel_Single_Test_LLM, self).__init__()
        self.g_type = list(config.g_type)

        self.use_text = config.use_text

        self._config_model = config.model

        self.g_type_model_ast = config.g_type_model.ast
        self.g_type_model_cfg = config.g_type_model.cfg
        self.g_type_model_pdg = config.g_type_model.pdg

        if 'ast' in self.g_type:
            self.ast_readout = GlobalAttention(nn.Linear(256, 1))
            if self.g_type_model_ast == 'GIN':

                self.ast_GIN = GIN(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    hidden_channels=1024, out_channels=256, num_layers=3,
                    dropout=0, act=F.leaky_relu)
            elif self.g_type_model_ast == 'TreeLSTM':

                self.ast_TreeLSTM_1 = treelstm.TreeLSTM(
                    self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size, 1024)
                self.ast_TreeLSTM_2 = treelstm.TreeLSTM(1024, 512)
                self.ast_TreeLSTM_3 = treelstm.TreeLSTM(512, 256)
            elif self.g_type_model_ast == 'GCN':

                self.ast_GNN_1 = GCNConv(
                    self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size, 1024)
                self.ast_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
                self.ast_GNN_2 = GCNConv(1024, 512)
                self.ast_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
                self.ast_GNN_3 = GCNConv(512, 256)
            elif self.g_type_model_ast == 'GAT':

                self.ast_GNN_1 = GATConv(
                    self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    1024, heads=3, concat=False)
                self.ast_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
                self.ast_GNN_2 = GATConv(1024, 512, heads=3, concat=False)
                self.ast_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
                self.ast_GNN_3 = GATConv(512, 256, heads=3, concat=False)
            elif self.g_type_model_ast == 'FILM':

                self.ast_film_1 = FiLMConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, act=F.leaky_relu)
                self.ast_pooling_film_1 = SAGPooling(1024, ratio=0.8)
                self.ast_film_2 = FiLMConv(in_channels=1024, out_channels=512, act=F.leaky_relu)
                self.ast_pooling_film_2 = SAGPooling(512, ratio=0.8)
                self.ast_film_3 = FiLMConv(in_channels=512, out_channels=256, act=F.leaky_relu)
            elif self.g_type_model_ast == 'GMM':

                self.ast_GMM_1 = GMMConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, dim=1, kernel_size=2)
                self.ast_pooling_GMM_1 = SAGPooling(1024, ratio=0.8)
                self.ast_GMM_2 = GMMConv(
                    in_channels=1024, out_channels=512, dim=1, kernel_size=2)
                self.ast_pooling_GMM_2 = SAGPooling(512, ratio=0.8)
                self.ast_GMM_3 = GMMConv(
                    in_channels=512, out_channels=256, dim=1, kernel_size=2)
            elif self.g_type_model_ast == 'TransformerConv':

                self.ast_TransformerConv_1 = TransformerConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, heads=5, concat=False)
                self.ast_pooling_TransformerConv_1 = SAGPooling(1024, ratio=0.8)
                self.ast_TransformerConv_2 = TransformerConv(
                    in_channels=1024, out_channels=512, heads=5, concat=False)
                self.ast_pooling_TransformerConv_2 = SAGPooling(512, ratio=0.8)
                self.ast_TransformerConv_3 = TransformerConv(
                    in_channels=512, out_channels=256, heads=5, concat=False)
            elif self.g_type_model_ast == 'TAG':

                self.ast_TAG_1 = TAGConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024)
                self.ast_pooling_TAG_1 = SAGPooling(1024, ratio=0.8)
                self.ast_TAG_2 = TAGConv(in_channels=1024, out_channels=512)
                self.ast_pooling_TAG_2 = SAGPooling(512, ratio=0.8)
                self.ast_TAG_3 = TAGConv(in_channels=512, out_channels=256)
            elif self.g_type_model_ast == 'ResGatedGraphConv':

                self.ast_ResGatedGraphConv_1 = ResGatedGraphConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024)
                self.ast_pooling_ResGatedGraphConv_1 = SAGPooling(1024, ratio=0.8)
                self.ast_ResGatedGraphConv_2 = ResGatedGraphConv(in_channels=1024, out_channels=512)
                self.ast_pooling_ResGatedGraphConv_2 = SAGPooling(512, ratio=0.8)
                self.ast_ResGatedGraphConv_3 = ResGatedGraphConv(in_channels=512, out_channels=256)
            elif self.g_type_model_ast == 'SAGE':

                self.ast_SAGE_1 = SAGEConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024)
                self.ast_pooling_SAGE_1 = SAGPooling(1024, ratio=0.8)
                self.ast_SAGE_2 = SAGEConv(in_channels=1024, out_channels=512)
                self.ast_pooling_SAGE_2 = SAGPooling(512, ratio=0.8)
                self.ast_SAGE_3 = SAGEConv(in_channels=512, out_channels=256)
            else:
                raise ValueError(f"Unsupported GNN type: {self.g_type_model_ast}")

        if 'cfg' in self.g_type:
            self.cfg_readout = GlobalAttention(nn.Linear(256, 1))
            if self.g_type_model_cfg == 'GIN':

                self.cfg_GIN = GIN(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    hidden_channels=1024, out_channels=256, num_layers=3,
                    dropout=0, act=F.leaky_relu)
            elif self.g_type_model_cfg == 'GCN':

                self.cfg_GNN_1 = GCNConv(
                    self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    1024)
                self.cfg_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
                self.cfg_GNN_2 = GCNConv(1024, 512)
                self.cfg_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
                self.cfg_GNN_3 = GCNConv(512, 256)
            elif self.g_type_model_cfg == 'GAT':

                self.cfg_GNN_1 = GATConv(
                    self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    1024, heads=3, concat=False)
                self.cfg_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
                self.cfg_GNN_2 = GATConv(1024, 512, heads=3, concat=False)
                self.cfg_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
                self.cfg_GNN_3 = GATConv(512, 256, heads=3, concat=False)
            elif self.g_type_model_cfg == 'FILM':

                self.cfg_film_1 = FiLMConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, act=F.leaky_relu)
                self.cfg_pooling_film_1 = SAGPooling(1024, ratio=0.8)
                self.cfg_film_2 = FiLMConv(in_channels=1024, out_channels=512, act=F.leaky_relu)
                self.cfg_pooling_film_2 = SAGPooling(512, ratio=0.8)
                self.cfg_film_3 = FiLMConv(in_channels=512, out_channels=256, act=F.leaky_relu)
            elif self.g_type_model_cfg == 'GMM':

                self.cfg_GMM_1 = GMMConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, dim=1, kernel_size=2)
                self.cfg_pooling_GMM_1 = SAGPooling(1024, ratio=0.8)
                self.cfg_GMM_2 = GMMConv(
                    in_channels=1024, out_channels=512, dim=1, kernel_size=2)
                self.cfg_pooling_GMM_2 = SAGPooling(512, ratio=0.8)
                self.cfg_GMM_3 = GMMConv(
                    in_channels=512, out_channels=256, dim=1, kernel_size=2)
            elif self.g_type_model_cfg == 'TransformerConv':

                self.cfg_TransformerConv_1 = TransformerConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, heads=5, concat=False)
                self.cfg_pooling_TransformerConv_1 = SAGPooling(1024, ratio=0.8)
                self.cfg_TransformerConv_2 = TransformerConv(
                    in_channels=1024, out_channels=512, heads=5, concat=False)
                self.cfg_pooling_TransformerConv_2 = SAGPooling(512, ratio=0.8)
                self.cfg_TransformerConv_3 = TransformerConv(
                    in_channels=512, out_channels=256, heads=5, concat=False)
            elif self.g_type_model_cfg == 'TAG':

                self.cfg_TAG_1 = TAGConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024)
                self.cfg_pooling_TAG_1 = SAGPooling(1024, ratio=0.8)
                self.cfg_TAG_2 = TAGConv(in_channels=1024, out_channels=512)
                self.cfg_pooling_TAG_2 = SAGPooling(512, ratio=0.8)
                self.cfg_TAG_3 = TAGConv(in_channels=512, out_channels=256)
            elif self.g_type_model_cfg == 'ResGatedGraphConv':

                self.cfg_ResGatedGraphConv_1 = ResGatedGraphConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024)
                self.cfg_pooling_ResGatedGraphConv_1 = SAGPooling(1024, ratio=0.8)
                self.cfg_ResGatedGraphConv_2 = ResGatedGraphConv(in_channels=1024, out_channels=512)
                self.cfg_pooling_ResGatedGraphConv_2 = SAGPooling(512, ratio=0.8)
                self.cfg_ResGatedGraphConv_3 = ResGatedGraphConv(in_channels=512, out_channels=256)
            elif self.g_type_model_cfg == 'SAGE':

                self.cfg_SAGE_1 = SAGEConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024)
                self.cfg_pooling_SAGE_1 = SAGPooling(1024, ratio=0.8)
                self.cfg_SAGE_2 = SAGEConv(in_channels=1024, out_channels=512)
                self.cfg_pooling_SAGE_2 = SAGPooling(512, ratio=0.8)
                self.cfg_SAGE_3 = SAGEConv(in_channels=512, out_channels=256)
            else:
                raise ValueError(f"Unsupported GNN type: {self.g_type_model_cfg}")

        if 'pdg' in self.g_type:
            self.pdg_readout = GlobalAttention(nn.Linear(256, 1))
            if self.g_type_model_pdg == 'GIN':

                self.pdg_GIN = GIN(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    hidden_channels=1024, out_channels=256, num_layers=3,
                    dropout=0, act=F.leaky_relu)
            elif self.g_type_model_pdg == 'GCN':

                self.pdg_GNN_1 = RGCNConv(
                    self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    1024, num_relations=2)
                self.pdg_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
                self.pdg_GNN_2 = RGCNConv(1024, 512, num_relations=2)
                self.pdg_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
                self.pdg_GNN_3 = RGCNConv(512, 256, num_relations=2)
            elif self.g_type_model_pdg == 'GAT':

                self.pdg_GNN_1 = GATConv(
                    self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    1024, heads=1, concat=False)
                self.pdg_pooling_GNN_1 = SAGPooling(1024, ratio=0.8)
                self.pdg_GNN_2 = GATConv(1024, 512, heads=1, concat=False)
                self.pdg_pooling_GNN_2 = SAGPooling(512, ratio=0.8)
                self.pdg_GNN_3 = GATConv(512, 256, heads=1, concat=False)
            elif self.g_type_model_pdg == 'FILM':

                self.pdg_film_1 = FiLMConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, num_relations=2, act=F.leaky_relu)
                self.pdg_pooling_film_1 = SAGPooling(1024, ratio=0.8)
                self.pdg_film_2 = FiLMConv(
                    in_channels=1024, out_channels=512, num_relations=2, act=F.leaky_relu)
                self.pdg_pooling_film_2 = SAGPooling(512, ratio=0.8)
                self.pdg_film_3 = FiLMConv(
                    in_channels=512, out_channels=256, num_relations=2, act=F.leaky_relu)
            elif self.g_type_model_pdg == 'GMM':

                self.pdg_GMM_1 = GMMConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, dim=1, kernel_size=2)
                self.pdg_pooling_GMM_1 = SAGPooling(1024, ratio=0.8)
                self.pdg_GMM_2 = GMMConv(
                    in_channels=1024, out_channels=512, dim=1, kernel_size=2)
                self.pdg_pooling_GMM_2 = SAGPooling(512, ratio=0.8)
                self.pdg_GMM_3 = GMMConv(
                    in_channels=512, out_channels=256, dim=1, kernel_size=2)
            elif self.g_type_model_pdg == 'TransformerConv':

                self.pdg_TransformerConv_1 = TransformerConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, heads=1, concat=False)
                self.pdg_pooling_TransformerConv_1 = SAGPooling(1024, ratio=0.8)
                self.pdg_TransformerConv_2 = TransformerConv(
                    in_channels=1024, out_channels=512, heads=1, concat=False)
                self.pdg_pooling_TransformerConv_2 = SAGPooling(512, ratio=0.8)
                self.pdg_TransformerConv_3 = TransformerConv(
                    in_channels=512, out_channels=256, heads=1, concat=False)
            elif self.g_type_model_pdg == 'TAG':

                self.pdg_TAG_1 = TAGConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024)
                self.pdg_pooling_TAG_1 = SAGPooling(1024, ratio=0.8)
                self.pdg_TAG_2 = TAGConv(in_channels=1024, out_channels=512)
                self.pdg_pooling_TAG_2 = SAGPooling(512, ratio=0.8)
                self.pdg_TAG_3 = TAGConv(in_channels=512, out_channels=256)
            elif self.g_type_model_pdg == 'ResGatedGraphConv':

                self.pdg_ResGatedGraphConv_1 = ResGatedGraphConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024, edge_dim=1)
                self.pdg_pooling_ResGatedGraphConv_1 = SAGPooling(1024, ratio=0.8)
                self.pdg_ResGatedGraphConv_2 = ResGatedGraphConv(in_channels=1024, out_channels=512, edge_dim=1)
                self.pdg_pooling_ResGatedGraphConv_2 = SAGPooling(512, ratio=0.8)
                self.pdg_ResGatedGraphConv_3 = ResGatedGraphConv(in_channels=512, out_channels=256, edge_dim=1)
            elif self.g_type_model_pdg == 'SAGE':

                self.pdg_SAGEConv_1 = SAGEConv(
                    in_channels=self._config_model.embedding[self._config_model.embedding.name_usevar].embed_size,
                    out_channels=1024)
                self.pdg_pooling_SAGEConv_1 = SAGPooling(1024, ratio=0.8)
                self.pdg_SAGEConv_2 = SAGEConv(in_channels=1024, out_channels=512)
                self.pdg_pooling_SAGEConv_2 = SAGPooling(512, ratio=0.8)
                self.pdg_SAGEConv_3 = SAGEConv(in_channels=512, out_channels=256)
            else:
                raise ValueError(f"Unsupported GNN type: {self.g_type_model_pdg}")

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

            if self.g_type_model_ast == 'TreeLSTM':

                node_order, edge_order = calculate_evaluation_orders_cuda(ast_edge_index.t(), len(ast_node_inf), device)
                ast_TreeLSTM_out_h_1, _ = self.ast_TreeLSTM_1(ast_node_embedding.float(), node_order,
                                                              ast_edge_index.t(),
                                                              edge_order)
                ast_TreeLSTM_out_h_2, _ = self.ast_TreeLSTM_2(ast_TreeLSTM_out_h_1, node_order, ast_edge_index.t(),
                                                              edge_order)
                ast_TreeLSTM_out_h_3, _ = self.ast_TreeLSTM_3(ast_TreeLSTM_out_h_2, node_order, ast_edge_index.t(),
                                                              edge_order)
                ast_readout = self.ast_readout(ast_TreeLSTM_out_h_3, ast_batch)
            elif self.g_type_model_ast == 'GIN':

                ast_node_embedding_GIN = self.ast_GIN(ast_node_embedding, ast_edge_index, batch=ast_batch,
                                                      batch_size=ast_batch.max().item() + 1)
                ast_readout = self.ast_readout(ast_node_embedding_GIN, ast_batch)
            elif self.g_type_model_ast == 'GCN' or self.g_type_model_ast == 'GAT':

                ast_node_embedding_GNN_1 = F.leaky_relu(self.ast_GNN_1(ast_node_embedding, ast_edge_index))
                ast_node_embedding_GNN_1, ast_edge_index_GNN_1, ast_edge_type_GNN_1, ast_batch_GNN_1, _, _ = self.ast_pooling_GNN_1(
                    ast_node_embedding_GNN_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
                ast_node_embedding_GNN_2 = F.leaky_relu(self.ast_GNN_2(ast_node_embedding_GNN_1, ast_edge_index_GNN_1))
                ast_node_embedding_GNN_2, ast_edge_index_GNN_2, ast_edge_type_GNN_2, ast_batch_GNN_2, _, _ = self.ast_pooling_GNN_2(
                    ast_node_embedding_GNN_2, ast_edge_index_GNN_1, edge_attr=ast_edge_type_GNN_1,
                    batch=ast_batch_GNN_1)
                ast_node_embedding_GNN_3 = F.leaky_relu(self.ast_GNN_3(ast_node_embedding_GNN_2, ast_edge_index_GNN_2))
                ast_readout = self.ast_readout(ast_node_embedding_GNN_3, ast_batch_GNN_2)
            elif self.g_type_model_ast == 'FILM':

                ast_node_embedding_FiLM_1 = self.ast_film_1(ast_node_embedding, ast_edge_index)
                ast_node_embedding_FiLM_1, ast_edge_index_FiLM_1, ast_edge_type_FiLM_1, ast_batch_FiLM_1, _, _ = self.ast_pooling_film_1(
                    ast_node_embedding_FiLM_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
                ast_node_embedding_FiLM_2 = self.ast_film_2(ast_node_embedding_FiLM_1, ast_edge_index_FiLM_1)
                ast_node_embedding_FiLM_2, ast_edge_index_FiLM_2, ast_edge_type_FiLM_2, ast_batch_FiLM_2, _, _ = self.ast_pooling_film_2(
                    ast_node_embedding_FiLM_2, ast_edge_index_FiLM_1, edge_attr=ast_edge_type_FiLM_1,
                    batch=ast_batch_FiLM_1)
                ast_node_embedding_FiLM_3 = self.ast_film_3(ast_node_embedding_FiLM_2, ast_edge_index_FiLM_2)
                ast_readout = self.ast_readout(ast_node_embedding_FiLM_3, ast_batch_FiLM_2)
            elif self.g_type_model_ast == 'GMM':

                ast_node_embedding_GMM_1 = self.ast_GMM_1(ast_node_embedding, ast_edge_index,
                                                          edge_attr=ast_edge_type.unsqueeze(1))
                ast_node_embedding_GMM_1, ast_edge_index_GMM_1, ast_edge_type_GMM_1, ast_batch_GMM_1, _, _ = self.ast_pooling_GMM_1(
                    ast_node_embedding_GMM_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
                ast_node_embedding_GMM_2 = self.ast_GMM_2(ast_node_embedding_GMM_1, ast_edge_index_GMM_1,
                                                          edge_attr=ast_edge_type_GMM_1.unsqueeze(1))
                ast_node_embedding_GMM_2, ast_edge_index_GMM_2, ast_edge_type_GMM_2, ast_batch_GMM_2, _, _ = self.ast_pooling_GMM_2(
                    ast_node_embedding_GMM_2, ast_edge_index_GMM_1, edge_attr=ast_edge_type_GMM_1,
                    batch=ast_batch_GMM_1)
                ast_node_embedding_GMM_3 = self.ast_GMM_3(ast_node_embedding_GMM_2, ast_edge_index_GMM_2,
                                                          edge_attr=ast_edge_type_GMM_2.unsqueeze(1))
                ast_readout = self.ast_readout(ast_node_embedding_GMM_3, ast_batch_GMM_2)
            elif self.g_type_model_ast == 'TransformerConv':

                ast_node_embedding_Transformer_1 = self.ast_TransformerConv_1(ast_node_embedding, ast_edge_index)
                ast_node_embedding_Transformer_1, ast_edge_index_Transformer_1, ast_edge_type_Transformer_1, ast_batch_Transformer_1, _, _ = self.ast_pooling_TransformerConv_1(
                    ast_node_embedding_Transformer_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
                ast_node_embedding_Transformer_2 = self.ast_TransformerConv_2(ast_node_embedding_Transformer_1,
                                                                              ast_edge_index_Transformer_1)
                ast_node_embedding_Transformer_2, ast_edge_index_Transformer_2, ast_edge_type_Transformer_2, ast_batch_Transformer_2, _, _ = self.ast_pooling_TransformerConv_2(
                    ast_node_embedding_Transformer_2, ast_edge_index_Transformer_1,
                    edge_attr=ast_edge_type_Transformer_1, batch=ast_batch_Transformer_1)
                ast_node_embedding_Transformer_3 = self.ast_TransformerConv_3(ast_node_embedding_Transformer_2,
                                                                              ast_edge_index_Transformer_2)
                ast_readout = self.ast_readout(ast_node_embedding_Transformer_3, ast_batch_Transformer_2)
            elif self.g_type_model_ast == 'TAG':

                ast_node_embedding_TAG_1 = self.ast_TAG_1(ast_node_embedding, ast_edge_index)
                ast_node_embedding_TAG_1, ast_edge_index_TAG_1, ast_edge_type_TAG_1, ast_batch_TAG_1, _, _ = self.ast_pooling_TAG_1(
                    ast_node_embedding_TAG_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
                ast_node_embedding_TAG_2 = self.ast_TAG_2(ast_node_embedding_TAG_1, ast_edge_index_TAG_1)
                ast_node_embedding_TAG_2, ast_edge_index_TAG_2, ast_edge_type_TAG_2, ast_batch_TAG_2, _, _ = self.ast_pooling_TAG_2(
                    ast_node_embedding_TAG_2, ast_edge_index_TAG_1, edge_attr=ast_edge_type_TAG_1,
                    batch=ast_batch_TAG_1)
                ast_node_embedding_TAG_3 = self.ast_TAG_3(ast_node_embedding_TAG_2, ast_edge_index_TAG_2)
                ast_readout = self.ast_readout(ast_node_embedding_TAG_3, ast_batch_TAG_2)
            elif self.g_type_model_ast == 'ResGatedGraphConv':

                ast_node_embedding_ResGatedGraphConv_1 = self.ast_ResGatedGraphConv_1(ast_node_embedding,
                                                                                      ast_edge_index)
                ast_node_embedding_ResGatedGraphConv_1, ast_edge_index_ResGatedGraphConv_1, ast_edge_type_ResGatedGraphConv_1, ast_batch_ResGatedGraphConv_1, _, _ = self.ast_pooling_ResGatedGraphConv_1(
                    ast_node_embedding_ResGatedGraphConv_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
                ast_node_embedding_ResGatedGraphConv_2 = self.ast_ResGatedGraphConv_2(
                    ast_node_embedding_ResGatedGraphConv_1, ast_edge_index_ResGatedGraphConv_1)
                ast_node_embedding_ResGatedGraphConv_2, ast_edge_index_ResGatedGraphConv_2, ast_edge_type_ResGatedGraphConv_2, ast_batch_ResGatedGraphConv_2, _, _ = self.ast_pooling_ResGatedGraphConv_2(
                    ast_node_embedding_ResGatedGraphConv_2, ast_edge_index_ResGatedGraphConv_1,
                    edge_attr=ast_edge_type_ResGatedGraphConv_1, batch=ast_batch_ResGatedGraphConv_1)
                ast_node_embedding_ResGatedGraphConv_3 = self.ast_ResGatedGraphConv_3(
                    ast_node_embedding_ResGatedGraphConv_2, ast_edge_index_ResGatedGraphConv_2)
                ast_readout = self.ast_readout(ast_node_embedding_ResGatedGraphConv_3, ast_batch_ResGatedGraphConv_2)
            elif self.g_type_model_ast == 'SAGE':

                ast_node_embedding_SAGEConv_1 = self.ast_SAGE_1(ast_node_embedding, ast_edge_index)
                ast_node_embedding_SAGEConv_1, ast_edge_index_SAGEConv_1, ast_edge_type_SAGEConv_1, ast_batch_SAGEConv_1, _, _ = self.ast_pooling_SAGE_1(
                    ast_node_embedding_SAGEConv_1, ast_edge_index, edge_attr=ast_edge_type, batch=ast_batch)
                ast_node_embedding_SAGEConv_2 = self.ast_SAGE_2(ast_node_embedding_SAGEConv_1,
                                                                ast_edge_index_SAGEConv_1)
                ast_node_embedding_SAGEConv_2, ast_edge_index_SAGEConv_2, ast_edge_type_SAGEConv_2, ast_batch_SAGEConv_2, _, _ = self.ast_pooling_SAGE_2(
                    ast_node_embedding_SAGEConv_2, ast_edge_index_SAGEConv_1, edge_attr=ast_edge_type_SAGEConv_1,
                    batch=ast_batch_SAGEConv_1)
                ast_node_embedding_SAGEConv_3 = self.ast_SAGE_3(ast_node_embedding_SAGEConv_2,
                                                                ast_edge_index_SAGEConv_2)
                ast_readout = self.ast_readout(ast_node_embedding_SAGEConv_3, ast_batch_SAGEConv_2)
            else:
                raise ValueError(f"Forward unsupported GNN type: {self.g_type_model_ast}")

        if 'cfg' in self.g_type:
            cfg_graph = data['cfg']
            cfg_node_inf = cfg_graph.x
            cfg_edge_index = cfg_graph.edge_index
            cfg_batch = cfg_graph.batch
            cfg_edge_type = cfg_graph.edge_type

            cfg_node_embedding = cfg_node_inf

            if self.g_type_model_cfg == 'GIN':

                cfg_node_embedding_GIN = self.cfg_GIN(cfg_node_embedding, cfg_edge_index, batch=cfg_batch,
                                                      batch_size=cfg_batch.max().item() + 1)
                cfg_readout = self.cfg_readout(cfg_node_embedding_GIN, cfg_batch)
            elif self.g_type_model_cfg == 'GCN' or self.g_type_model_cfg == 'GAT':

                cfg_node_embedding_GNN_1 = F.leaky_relu(self.cfg_GNN_1(cfg_node_embedding, cfg_edge_index))

                cfg_node_embedding_GNN_1, cfg_edge_index_GNN_1, cfg_edge_type_GNN_1, cfg_batch_GNN_1, _, _ = self.cfg_pooling_GNN_1(
                    cfg_node_embedding_GNN_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
                cfg_node_embedding_GNN_2 = F.leaky_relu(self.cfg_GNN_2(cfg_node_embedding_GNN_1, cfg_edge_index_GNN_1))

                cfg_node_embedding_GNN_2, cfg_edge_index_GNN_2, cfg_edge_type_GNN_2, cfg_batch_GNN_2, _, _ = self.cfg_pooling_GNN_2(
                    cfg_node_embedding_GNN_2, cfg_edge_index_GNN_1, edge_attr=cfg_edge_type_GNN_1,
                    batch=cfg_batch_GNN_1)
                cfg_node_embedding_GNN_3 = F.leaky_relu(self.cfg_GNN_3(cfg_node_embedding_GNN_2, cfg_edge_index_GNN_2))
                cfg_readout = self.cfg_readout(cfg_node_embedding_GNN_3, cfg_batch_GNN_2)
            elif self.g_type_model_cfg == 'FILM':

                cfg_node_embedding_FiLM_1 = self.cfg_film_1(cfg_node_embedding, cfg_edge_index)
                cfg_node_embedding_FiLM_1, cfg_edge_index_FiLM_1, cfg_edge_type_FiLM_1, cfg_batch_FiLM_1, _, _ = self.cfg_pooling_film_1(
                    cfg_node_embedding_FiLM_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
                cfg_node_embedding_FiLM_2 = self.cfg_film_2(cfg_node_embedding_FiLM_1, cfg_edge_index_FiLM_1)
                cfg_node_embedding_FiLM_2, cfg_edge_index_FiLM_2, cfg_edge_type_FiLM_2, cfg_batch_FiLM_2, _, _ = self.cfg_pooling_film_2(
                    cfg_node_embedding_FiLM_2, cfg_edge_index_FiLM_1, edge_attr=cfg_edge_type_FiLM_1,
                    batch=cfg_batch_FiLM_1)
                cfg_node_embedding_FiLM_3 = self.cfg_film_3(cfg_node_embedding_FiLM_2, cfg_edge_index_FiLM_2)
                cfg_readout = self.cfg_readout(cfg_node_embedding_FiLM_3, cfg_batch_FiLM_2)
            elif self.g_type_model_cfg == 'GMM':

                cfg_node_embedding_GMM_1 = self.cfg_GMM_1(cfg_node_embedding, cfg_edge_index,
                                                          edge_attr=cfg_edge_type.unsqueeze(1))
                cfg_node_embedding_GMM_1, cfg_edge_index_GMM_1, cfg_edge_type_GMM_1, cfg_batch_GMM_1, _, _ = self.cfg_pooling_GMM_1(
                    cfg_node_embedding_GMM_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
                cfg_node_embedding_GMM_2 = self.cfg_GMM_2(cfg_node_embedding_GMM_1, cfg_edge_index_GMM_1,
                                                          edge_attr=cfg_edge_type_GMM_1.unsqueeze(1))
                cfg_node_embedding_GMM_2, cfg_edge_index_GMM_2, cfg_edge_type_GMM_2, cfg_batch_GMM_2, _, _ = self.cfg_pooling_GMM_2(
                    cfg_node_embedding_GMM_2, cfg_edge_index_GMM_1, edge_attr=cfg_edge_type_GMM_1,
                    batch=cfg_batch_GMM_1)
                cfg_node_embedding_GMM_3 = self.cfg_GMM_3(cfg_node_embedding_GMM_2, cfg_edge_index_GMM_2,
                                                          edge_attr=cfg_edge_type_GMM_2.unsqueeze(1))
                cfg_readout = self.cfg_readout(cfg_node_embedding_GMM_3, cfg_batch_GMM_2)
            elif self.g_type_model_cfg == 'TransformerConv':

                cfg_node_embedding_Transformer_1 = self.cfg_TransformerConv_1(cfg_node_embedding, cfg_edge_index)
                cfg_node_embedding_Transformer_1, cfg_edge_index_Transformer_1, cfg_edge_type_Transformer_1, cfg_batch_Transformer_1, _, _ = self.cfg_pooling_TransformerConv_1(
                    cfg_node_embedding_Transformer_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
                cfg_node_embedding_Transformer_2 = self.cfg_TransformerConv_2(cfg_node_embedding_Transformer_1,
                                                                              cfg_edge_index_Transformer_1)
                cfg_node_embedding_Transformer_2, cfg_edge_index_Transformer_2, cfg_edge_type_Transformer_2, cfg_batch_Transformer_2, _, _ = self.cfg_pooling_TransformerConv_2(
                    cfg_node_embedding_Transformer_2, cfg_edge_index_Transformer_1,
                    edge_attr=cfg_edge_type_Transformer_1, batch=cfg_batch_Transformer_1)
                cfg_node_embedding_Transformer_3 = self.cfg_TransformerConv_3(cfg_node_embedding_Transformer_2,
                                                                              cfg_edge_index_Transformer_2)
                cfg_readout = self.cfg_readout(cfg_node_embedding_Transformer_3, cfg_batch_Transformer_2)
            elif self.g_type_model_cfg == 'TAG':

                cfg_node_embedding_TAG_1 = self.cfg_TAG_1(cfg_node_embedding, cfg_edge_index)
                cfg_node_embedding_TAG_1, cfg_edge_index_TAG_1, cfg_edge_type_TAG_1, cfg_batch_TAG_1, _, _ = self.cfg_pooling_TAG_1(
                    cfg_node_embedding_TAG_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
                cfg_node_embedding_TAG_2 = self.cfg_TAG_2(cfg_node_embedding_TAG_1, cfg_edge_index_TAG_1)
                cfg_node_embedding_TAG_2, cfg_edge_index_TAG_2, cfg_edge_type_TAG_2, cfg_batch_TAG_2, _, _ = self.cfg_pooling_TAG_2(
                    cfg_node_embedding_TAG_2, cfg_edge_index_TAG_1, edge_attr=cfg_edge_type_TAG_1,
                    batch=cfg_batch_TAG_1)
                cfg_node_embedding_TAG_3 = self.cfg_TAG_3(cfg_node_embedding_TAG_2, cfg_edge_index_TAG_2)
                cfg_readout = self.cfg_readout(cfg_node_embedding_TAG_3, cfg_batch_TAG_2)
            elif self.g_type_model_cfg == 'ResGatedGraphConv':

                cfg_node_embedding_ResGatedGraphConv_1 = self.cfg_ResGatedGraphConv_1(cfg_node_embedding,
                                                                                      cfg_edge_index)
                cfg_node_embedding_ResGatedGraphConv_1, cfg_edge_index_ResGatedGraphConv_1, cfg_edge_type_ResGatedGraphConv_1, cfg_batch_ResGatedGraphConv_1, _, _ = self.cfg_pooling_ResGatedGraphConv_1(
                    cfg_node_embedding_ResGatedGraphConv_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
                cfg_node_embedding_ResGatedGraphConv_2 = self.cfg_ResGatedGraphConv_2(
                    cfg_node_embedding_ResGatedGraphConv_1, cfg_edge_index_ResGatedGraphConv_1)
                cfg_node_embedding_ResGatedGraphConv_2, cfg_edge_index_ResGatedGraphConv_2, cfg_edge_type_ResGatedGraphConv_2, cfg_batch_ResGatedGraphConv_2, _, _ = self.cfg_pooling_ResGatedGraphConv_2(
                    cfg_node_embedding_ResGatedGraphConv_2, cfg_edge_index_ResGatedGraphConv_1,
                    edge_attr=cfg_edge_type_ResGatedGraphConv_1, batch=cfg_batch_ResGatedGraphConv_1)
                cfg_node_embedding_ResGatedGraphConv_3 = self.cfg_ResGatedGraphConv_3(
                    cfg_node_embedding_ResGatedGraphConv_2, cfg_edge_index_ResGatedGraphConv_2)
                cfg_readout = self.cfg_readout(cfg_node_embedding_ResGatedGraphConv_3, cfg_batch_ResGatedGraphConv_2)
            elif self.g_type_model_cfg == 'SAGE':

                cfg_node_embedding_SAGEConv_1 = self.cfg_SAGE_1(cfg_node_embedding, cfg_edge_index)
                cfg_node_embedding_SAGEConv_1, cfg_edge_index_SAGEConv_1, cfg_edge_type_SAGEConv_1, cfg_batch_SAGEConv_1, _, _ = self.cfg_pooling_SAGE_1(
                    cfg_node_embedding_SAGEConv_1, cfg_edge_index, edge_attr=cfg_edge_type, batch=cfg_batch)
                cfg_node_embedding_SAGEConv_2 = self.cfg_SAGE_2(cfg_node_embedding_SAGEConv_1,
                                                                cfg_edge_index_SAGEConv_1)
                cfg_node_embedding_SAGEConv_2, cfg_edge_index_SAGEConv_2, cfg_edge_type_SAGEConv_2, cfg_batch_SAGEConv_2, _, _ = self.cfg_pooling_SAGE_2(
                    cfg_node_embedding_SAGEConv_2, cfg_edge_index_SAGEConv_1, edge_attr=cfg_edge_type_SAGEConv_1,
                    batch=cfg_batch_SAGEConv_1)
                cfg_node_embedding_SAGEConv_3 = self.cfg_SAGE_3(cfg_node_embedding_SAGEConv_2,
                                                                cfg_edge_index_SAGEConv_2)
                cfg_readout = self.cfg_readout(cfg_node_embedding_SAGEConv_3, cfg_batch_SAGEConv_2)
            else:
                raise ValueError(f"Forward unsupported GNN type: {self.g_type_model_cfg}")

        if 'pdg' in self.g_type:
            pdg_graph = data['pdg']
            pdg_node_inf = pdg_graph.x
            pdg_edge_index = pdg_graph.edge_index
            pdg_batch = pdg_graph.batch
            pdg_edge_type = pdg_graph.edge_type
            pdg_edge_type = self.reset_edge_type(pdg_edge_type, device)

            pdg_node_embedding = pdg_node_inf

            if self.g_type_model_pdg == 'GIN':

                pdg_node_embedding_GIN = self.pdg_GIN(pdg_node_embedding, pdg_edge_index, batch=pdg_batch,
                                                      batch_size=pdg_batch.max().item() + 1)
                pdg_readout = self.pdg_readout(pdg_node_embedding_GIN, pdg_batch)
            elif self.g_type_model_pdg == 'GCN':

                pdg_node_embedding_GNN_1 = F.leaky_relu(
                    self.pdg_GNN_1(pdg_node_embedding, pdg_edge_index, edge_type=pdg_edge_type))

                pdg_node_embedding_GNN_1, pdg_edge_index_GNN_1, pdg_edge_type_GNN_1, pdg_batch_GNN_1, _, _ = self.pdg_pooling_GNN_1(
                    pdg_node_embedding_GNN_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
                pdg_node_embedding_GNN_2 = F.leaky_relu(
                    self.pdg_GNN_2(pdg_node_embedding_GNN_1, pdg_edge_index_GNN_1, edge_type=pdg_edge_type_GNN_1))

                pdg_node_embedding_GNN_2, pdg_edge_index_GNN_2, pdg_edge_type_GNN_2, pdg_batch_GNN_2, _, _ = self.pdg_pooling_GNN_2(
                    pdg_node_embedding_GNN_2, pdg_edge_index_GNN_1, edge_attr=pdg_edge_type_GNN_1,
                    batch=pdg_batch_GNN_1)
                pdg_node_embedding_GNN_3 = F.leaky_relu(
                    self.pdg_GNN_3(pdg_node_embedding_GNN_2, pdg_edge_index_GNN_2, edge_type=pdg_edge_type_GNN_2))
                pdg_readout = self.pdg_readout(pdg_node_embedding_GNN_3, pdg_batch_GNN_2)
            elif self.g_type_model_pdg == 'GAT':

                pdg_node_embedding_GNN_1 = F.leaky_relu(
                    self.pdg_GNN_1(pdg_node_embedding, pdg_edge_index))
                pdg_node_embedding_GNN_1, pdg_edge_index_GNN_1, pdg_edge_type_GNN_1, pdg_batch_GNN_1, _, _ = self.pdg_pooling_GNN_1(
                    pdg_node_embedding_GNN_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
                pdg_node_embedding_GNN_2 = F.leaky_relu(
                    self.pdg_GNN_2(pdg_node_embedding_GNN_1, pdg_edge_index_GNN_1))
                pdg_node_embedding_GNN_2, pdg_edge_index_GNN_2, pdg_edge_type_GNN_2, pdg_batch_GNN_2, _, _ = self.pdg_pooling_GNN_2(
                    pdg_node_embedding_GNN_2, pdg_edge_index_GNN_1, edge_attr=pdg_edge_type_GNN_1,
                    batch=pdg_batch_GNN_1)
                pdg_node_embedding_GNN_3 = F.leaky_relu(
                    self.pdg_GNN_3(pdg_node_embedding_GNN_2, pdg_edge_index_GNN_2))
                pdg_readout = self.pdg_readout(pdg_node_embedding_GNN_3, pdg_batch_GNN_2)
            elif self.g_type_model_pdg == 'FILM':

                pdg_node_embedding_FiLM_1 = self.pdg_film_1(pdg_node_embedding, pdg_edge_index, edge_type=pdg_edge_type)

                pdg_node_embedding_FiLM_1, pdg_edge_index_FiLM_1, pdg_edge_type_FiLM_1, pdg_batch_FiLM_1, _, _ = self.pdg_pooling_film_1(
                    pdg_node_embedding_FiLM_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
                pdg_node_embedding_FiLM_2 = self.pdg_film_2(pdg_node_embedding_FiLM_1, pdg_edge_index_FiLM_1,
                                                            edge_type=pdg_edge_type_FiLM_1)
                pdg_node_embedding_FiLM_2, pdg_edge_index_FiLM_2, pdg_edge_type_FiLM_2, pdg_batch_FiLM_2, _, _ = self.pdg_pooling_film_2(
                    pdg_node_embedding_FiLM_2, pdg_edge_index_FiLM_1, edge_attr=pdg_edge_type_FiLM_1,
                    batch=pdg_batch_FiLM_1)
                pdg_node_embedding_FiLM_3 = self.pdg_film_3(pdg_node_embedding_FiLM_2, pdg_edge_index_FiLM_2,
                                                            edge_type=pdg_edge_type_FiLM_2)
                pdg_readout = self.pdg_readout(pdg_node_embedding_FiLM_3, pdg_batch_FiLM_2)
            elif self.g_type_model_pdg == 'GMM':

                pdg_node_embedding_GMM_1 = self.pdg_GMM_1(pdg_node_embedding, pdg_edge_index,
                                                          edge_attr=pdg_edge_type.unsqueeze(1))
                pdg_node_embedding_GMM_1, pdg_edge_index_GMM_1, pdg_edge_type_GMM_1, pdg_batch_GMM_1, _, _ = self.pdg_pooling_GMM_1(
                    pdg_node_embedding_GMM_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
                pdg_node_embedding_GMM_2 = self.pdg_GMM_2(pdg_node_embedding_GMM_1, pdg_edge_index_GMM_1,
                                                          edge_attr=pdg_edge_type_GMM_1.unsqueeze(1))
                pdg_node_embedding_GMM_2, pdg_edge_index_GMM_2, pdg_edge_type_GMM_2, pdg_batch_GMM_2, _, _ = self.pdg_pooling_GMM_2(
                    pdg_node_embedding_GMM_2, pdg_edge_index_GMM_1, edge_attr=pdg_edge_type_GMM_1,
                    batch=pdg_batch_GMM_1)
                pdg_node_embedding_GMM_3 = self.pdg_GMM_3(pdg_node_embedding_GMM_2, pdg_edge_index_GMM_2,
                                                          edge_attr=pdg_edge_type_GMM_2.unsqueeze(1))
                pdg_readout = self.pdg_readout(pdg_node_embedding_GMM_3, pdg_batch_GMM_2)
            elif self.g_type_model_pdg == 'TransformerConv':

                pdg_node_embedding_Transformer_1 = self.pdg_TransformerConv_1(pdg_node_embedding, pdg_edge_index)
                pdg_node_embedding_Transformer_1, pdg_edge_index_Transformer_1, pdg_edge_type_Transformer_1, pdg_batch_Transformer_1, _, _ = self.pdg_pooling_TransformerConv_1(
                    pdg_node_embedding_Transformer_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
                pdg_node_embedding_Transformer_2 = self.pdg_TransformerConv_2(pdg_node_embedding_Transformer_1,
                                                                              pdg_edge_index_Transformer_1)
                pdg_node_embedding_Transformer_2, pdg_edge_index_Transformer_2, pdg_edge_type_Transformer_2, pdg_batch_Transformer_2, _, _ = self.pdg_pooling_TransformerConv_2(
                    pdg_node_embedding_Transformer_2, pdg_edge_index_Transformer_1,
                    edge_attr=pdg_edge_type_Transformer_1, batch=pdg_batch_Transformer_1)
                pdg_node_embedding_Transformer_3 = self.pdg_TransformerConv_3(pdg_node_embedding_Transformer_2,
                                                                              pdg_edge_index_Transformer_2)
                pdg_readout = self.pdg_readout(pdg_node_embedding_Transformer_3, pdg_batch_Transformer_2)
            elif self.g_type_model_pdg == 'TAG':

                pdg_node_embedding_TAG_1 = self.pdg_TAG_1(pdg_node_embedding, pdg_edge_index)
                pdg_node_embedding_TAG_1, pdg_edge_index_TAG_1, pdg_edge_type_TAG_1, pdg_batch_TAG_1, _, _ = self.pdg_pooling_TAG_1(
                    pdg_node_embedding_TAG_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
                pdg_node_embedding_TAG_2 = self.pdg_TAG_2(pdg_node_embedding_TAG_1, pdg_edge_index_TAG_1)
                pdg_node_embedding_TAG_2, pdg_edge_index_TAG_2, pdg_edge_type_TAG_2, pdg_batch_TAG_2, _, _ = self.pdg_pooling_TAG_2(
                    pdg_node_embedding_TAG_2, pdg_edge_index_TAG_1, edge_attr=pdg_edge_type_TAG_1,
                    batch=pdg_batch_TAG_1)
                pdg_node_embedding_TAG_3 = self.pdg_TAG_3(pdg_node_embedding_TAG_2, pdg_edge_index_TAG_2)
                pdg_readout = self.pdg_readout(pdg_node_embedding_TAG_3, pdg_batch_TAG_2)
            elif self.g_type_model_pdg == 'ResGatedGraphConv':

                pdg_node_embedding_ResGatedGraphConv_1 = self.pdg_ResGatedGraphConv_1(pdg_node_embedding,
                                                                                      pdg_edge_index,
                                                                                      edge_attr=pdg_edge_type.unsqueeze(
                                                                                          1))
                pdg_node_embedding_ResGatedGraphConv_1, pdg_edge_index_ResGatedGraphConv_1, pdg_edge_type_ResGatedGraphConv_1, pdg_batch_ResGatedGraphConv_1, _, _ = self.pdg_pooling_ResGatedGraphConv_1(
                    pdg_node_embedding_ResGatedGraphConv_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
                pdg_node_embedding_ResGatedGraphConv_2 = self.pdg_ResGatedGraphConv_2(
                    pdg_node_embedding_ResGatedGraphConv_1, pdg_edge_index_ResGatedGraphConv_1,
                    edge_attr=pdg_edge_type_ResGatedGraphConv_1.unsqueeze(1))
                pdg_node_embedding_ResGatedGraphConv_2, pdg_edge_index_ResGatedGraphConv_2, pdg_edge_type_ResGatedGraphConv_2, pdg_batch_ResGatedGraphConv_2, _, _ = self.pdg_pooling_ResGatedGraphConv_2(
                    pdg_node_embedding_ResGatedGraphConv_2, pdg_edge_index_ResGatedGraphConv_1,
                    edge_attr=pdg_edge_type_ResGatedGraphConv_1, batch=pdg_batch_ResGatedGraphConv_1)
                pdg_node_embedding_ResGatedGraphConv_3 = self.pdg_ResGatedGraphConv_3(
                    pdg_node_embedding_ResGatedGraphConv_2, pdg_edge_index_ResGatedGraphConv_2,
                    edge_attr=pdg_edge_type_ResGatedGraphConv_2.unsqueeze(1))
                pdg_readout = self.pdg_readout(pdg_node_embedding_ResGatedGraphConv_3, pdg_batch_ResGatedGraphConv_2)
            elif self.g_type_model_pdg == 'SAGE':

                pdg_node_embedding_SAGEConv_1 = self.pdg_SAGEConv_1(pdg_node_embedding, pdg_edge_index)
                pdg_node_embedding_SAGEConv_1, pdg_edge_index_SAGEConv_1, pdg_edge_type_SAGEConv_1, pdg_batch_SAGEConv_1, _, _ = self.pdg_pooling_SAGEConv_1(
                    pdg_node_embedding_SAGEConv_1, pdg_edge_index, edge_attr=pdg_edge_type, batch=pdg_batch)
                pdg_node_embedding_SAGEConv_2 = self.pdg_SAGEConv_2(pdg_node_embedding_SAGEConv_1,
                                                                    pdg_edge_index_SAGEConv_1)
                pdg_node_embedding_SAGEConv_2, pdg_edge_index_SAGEConv_2, pdg_edge_type_SAGEConv_2, pdg_batch_SAGEConv_2, _, _ = self.pdg_pooling_SAGEConv_2(
                    pdg_node_embedding_SAGEConv_2, pdg_edge_index_SAGEConv_1, edge_attr=pdg_edge_type_SAGEConv_1,
                    batch=pdg_batch_SAGEConv_1)
                pdg_node_embedding_SAGEConv_3 = self.pdg_SAGEConv_3(pdg_node_embedding_SAGEConv_2,
                                                                    pdg_edge_index_SAGEConv_2)
                pdg_readout = self.pdg_readout(pdg_node_embedding_SAGEConv_3, pdg_batch_SAGEConv_2)
            else:
                raise ValueError(f"Forward unsupported GNN type: {self.g_type_model_pdg}")

        readout_tensors = [t for t in [ast_readout, cfg_readout, pdg_readout] if t is not None]

        connectOut = torch.cat(readout_tensors, 1)

        if self.use_text:
            LLM_embeddings = self.LLM_Linear(source_code)
            connectOut = torch.cat((connectOut, LLM_embeddings), 1)

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
            self.LLM_Linear = nn.Linear(
                self._config_model.embedding[self._config_model.embedding.name_novar].embed_size, 256)

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

            ast_cfg_pdg_F = [
                nn.Linear(256, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU()
            ]
            self.ast_attention_F = nn.Sequential(*copy.deepcopy(ast_cfg_pdg_F))
            self.cfg_attention_F = nn.Sequential(*copy.deepcopy(ast_cfg_pdg_F))
            self.pdg_attention_F = nn.Sequential(*copy.deepcopy(ast_cfg_pdg_F))

            self.ast_attention = nn.Sequential(
                nn.Linear(MLP_input_size, 1024),
                nn.Sigmoid()
            )
            self.cfg_attention = nn.Sequential(
                nn.Linear(MLP_input_size, 1024),
                nn.Sigmoid()
            )
            self.pdg_attention = nn.Sequential(
                nn.Linear(MLP_input_size, 1024),
                nn.Sigmoid()
            )

            if self.use_text:
                self.text_attention_F = nn.Linear(256, 1024)
                self.text_attention = nn.Sequential(
                    nn.Linear(MLP_input_size, 1024),
                    nn.Sigmoid()
                )

            MLP_layers = [
                nn.Linear(MLP_input_size * 4, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
                nn.Dropout(0.1),

                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            ]
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

        connectOut = torch.cat(readout_tensors, 1)

        if self.use_text:
            LLM_embeddings = self.LLM_Linear(source_code)
            connectOut = torch.cat((connectOut, LLM_embeddings), 1)

        if self._config_pre_train_structure.used:
            ast_attention = self.ast_attention(connectOut)
            cfg_attention = self.cfg_attention(connectOut)
            pdg_attention = self.pdg_attention(connectOut)

            ast_attention_F = self.ast_attention_F(ast_readout)
            cfg_attention_F = self.cfg_attention_F(cfg_readout)
            pdg_attention_F = self.pdg_attention_F(pdg_readout)

            all_att_out = torch.cat((ast_attention * ast_attention_F,
                                     cfg_attention * cfg_attention_F,
                                     pdg_attention * pdg_attention_F), 1)

            if self.use_text:
                text_attention = self.text_attention(connectOut)
                text_attention_F = self.text_attention_F(LLM_embeddings)
                all_att_out = torch.cat((all_att_out, text_attention * text_attention_F), 1)

            connectOut = all_att_out

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
