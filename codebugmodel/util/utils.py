import os
import pickle

import torch
from omegaconf import DictConfig
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import networkx as nx
from transformers import (AutoConfig, AutoModel,
                          T5EncoderModel, AutoTokenizer, AutoModelForSequenceClassification)


def is_pyg_or_nx(G):
    if isinstance(G, Data):
        return 'pyg.Data'
    elif isinstance(G, nx.DiGraph):
        return 'networkx.DiGraph'
    else:
        return None


def assign_value_to_property(data):
    if data.y is None:
        data.y = data.stores[0]['graph_label']
    if data.x is None:
        data.x = data.stores[0]['label']
    return data


def load_pickle_form_pyg_data(file_path):
    with open(file_path, 'rb') as f:
        G = pickle.load(f)

    string_type = is_pyg_or_nx(G)
    if string_type == 'pyg.Data':
        return G
    elif string_type == 'networkx.DiGraph':

        if 'cwe' in G.graph:
            G.graph['cwe'] = {"cwe": G.graph['cwe']}
        return assign_value_to_property(from_networkx(G))
    else:
        return None


def save_pyg_data_to_pickle(G, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(G, f)


def find_matching_bracket(text, start_index, bracket_type=1):
    start_bracket = '{'
    end_bracket = '}'
    if bracket_type == 1:
        start_bracket = '{'
        end_bracket = '}'
    elif bracket_type == 2:
        start_bracket = '('
        end_bracket = ')'
    elif bracket_type == 3:
        start_bracket = '['
        end_bracket = ']'
    elif bracket_type == 4:
        start_bracket = '<'
        end_bracket = '>'
    else:
        return -1

    stack = 0
    for i in range(start_index, len(text)):
        if text[i] == start_bracket:
            stack += 1
        elif text[i] == end_bracket:
            stack -= 1
            if stack == 0:
                return i
    return -1


EMBEDDING_MODEL_CLASSES = {
    'auto': (AutoConfig, AutoModel, AutoTokenizer),

    't5': (AutoConfig, T5EncoderModel, AutoTokenizer),
}


def loadBertModel(configs: DictConfig):
    embed_config = configs.model.embedding

    name_novar = embed_config.name_novar
    name_usevar = embed_config.name_usevar

    current_model_config_usevar = embed_config[name_usevar]
    current_model_config_novar = embed_config[name_novar]

    model_type_usevar = current_model_config_usevar.model_type
    model_type_novar = current_model_config_novar.model_type

    config_class_usevar, model_class_usevar, tokenizer_class_usevar = EMBEDDING_MODEL_CLASSES[model_type_usevar]
    config_class_novar, model_class_novar, tokenizer_class_novar = EMBEDDING_MODEL_CLASSES[model_type_novar]

    cache_dir_usevar = current_model_config_usevar.cache_dir
    cache_dir_novar = current_model_config_novar.cache_dir

    checkpoint_prefix = 'checkpoint-best-acc/model.bin'

    path_dir_usevar = current_model_config_usevar.path_dir
    config_usevar = config_class_usevar.from_pretrained(path_dir_usevar, cache_dir=cache_dir_usevar,
                                                        local_files_only=True)
    config_usevar.num_labels = 1

    tokenizer_usevar = tokenizer_class_usevar.from_pretrained(path_dir_usevar, do_lower_case=False,
                                                              cache_dir=cache_dir_usevar,
                                                              local_files_only=True)
    model_usevar = model_class_usevar.from_pretrained(path_dir_usevar, from_tf=False,
                                                      config=config_usevar, cache_dir=cache_dir_usevar,
                                                      local_files_only=True)
    if current_model_config_usevar.fine_tuning:
        fine_tuning_path_usevar = current_model_config_usevar.fine_tuning_path_usevar
        old_model_state_dict_usevar = torch.load(os.path.join(fine_tuning_path_usevar, '{}'.format(checkpoint_prefix)))

        if model_type_usevar == 'auto':
            new_state_dict_usevar = convert_MyAutoModel_to_AutoModel(old_model_state_dict_usevar)
        elif model_type_usevar == 't5':
            new_state_dict_usevar = convert_MyAutoModel_to_T5EncoderModel(old_model_state_dict_usevar)
        else:
            raise ValueError('model_type_usevar error!')

        model_usevar.load_state_dict(new_state_dict_usevar, strict=False)

    path_dir_novar = current_model_config_novar.path_dir
    config_novar = config_class_novar.from_pretrained(path_dir_novar, cache_dir=cache_dir_novar,
                                                      local_files_only=True)
    config_novar.num_labels = 1

    tokenizer_novar = tokenizer_class_novar.from_pretrained(path_dir_novar, do_lower_case=False,
                                                            cache_dir=cache_dir_novar,
                                                            local_files_only=True)
    model_novar = model_class_novar.from_pretrained(path_dir_novar, from_tf=False,
                                                    config=config_novar, cache_dir=cache_dir_novar,
                                                    local_files_only=True)
    if current_model_config_novar.fine_tuning:
        fine_tuning_path_novar = current_model_config_novar.fine_tuning_path_novar
        old_model_state_dict_novar = torch.load(os.path.join(fine_tuning_path_novar, '{}'.format(checkpoint_prefix)))

        if model_type_novar == 'auto':
            new_state_dict_novar = convert_MyAutoModel_to_AutoModel(old_model_state_dict_novar)
        elif model_type_novar == 't5':
            new_state_dict_novar = convert_MyAutoModel_to_T5EncoderModel(old_model_state_dict_novar)
        else:
            raise ValueError('model_type_novar error!')

        model_novar.load_state_dict(new_state_dict_novar, strict=False)

    return config_usevar, tokenizer_usevar, model_usevar, config_novar, tokenizer_novar, model_novar


def convert_MyAutoModel_to_AutoModelForSequenceClassification(old_model_state_dict):
    new_model_state_dict = {}
    for key in old_model_state_dict.keys():
        new_key = key.replace('encoder.roberta.', 'roberta.').replace('encoder.classifier.', 'classifier.')
        new_model_state_dict[new_key] = old_model_state_dict[key]
    return new_model_state_dict


def convert_MyAutoModel_to_AutoModel(old_model_state_dict):
    new_model_state_dict = {}
    for key in old_model_state_dict.keys():
        new_key = key.replace('encoder.roberta.embeddings.', 'embeddings.').replace('encoder.roberta.encoder.',
                                                                                    'encoder.')

        if new_key.startswith('encoder.classifier'):
            continue
        new_model_state_dict[new_key] = old_model_state_dict[key]
    return new_model_state_dict


def convert_MyAutoModel_to_T5EncoderModel(old_model_state_dict):
    new_model_state_dict = {}
    for key in old_model_state_dict.keys():
        new_key = key.replace('encoder.transformer.shared.',
                              'shared.').replace('encoder.transformer.encoder.',
                                                 'encoder.').replace('encoder.transformer.decoder.', 'decoder.')

        if new_key.startswith('decoder') or new_key.startswith('encoder.classification_head'):
            continue
        new_model_state_dict[new_key] = old_model_state_dict[key]
    return new_model_state_dict


def calculate_evaluation_orders_cuda(adjacency_list, tree_size, device):
    '''Calculates the node_order and edge_order from a tree adjacency_list and the tree_size.

    The TreeLSTM model requires node_order and edge_order to be passed into the model along
    with the node features and adjacency_list.  We pre-calculate these orders as a speed
    optimization.
    '''
    adjacency_list = torch.tensor(adjacency_list, dtype=torch.long, device=device)

    node_ids = torch.arange(tree_size, device=device, dtype=torch.long)

    node_order = torch.zeros(tree_size, dtype=torch.long, device=device)
    unevaluated_nodes = torch.ones(tree_size, dtype=torch.bool, device=device)

    parent_nodes = adjacency_list[:, 0]
    child_nodes = adjacency_list[:, 1]

    old_unready_parents_len = None
    n = 0
    while unevaluated_nodes.any():

        unevaluated_mask = unevaluated_nodes[child_nodes]

        unready_parents = parent_nodes[unevaluated_mask]

        nodes_to_evaluate = unevaluated_nodes & ~torch.any(torch.eq(node_ids.unsqueeze(1), unready_parents), dim=1)

        if old_unready_parents_len == unready_parents.size(0):
            max_unready_child = torch.max(child_nodes[unevaluated_mask])
            nodes_to_evaluate[max_unready_child] = True
            n -= 1

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        old_unready_parents_len = unready_parents.size(0)

        n += 1

    edge_order = node_order[parent_nodes]

    return node_order, edge_order
