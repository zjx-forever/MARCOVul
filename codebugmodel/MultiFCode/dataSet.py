import gc
import json
import sys
import threading
from datetime import datetime
import shutil

import pytz
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import os
from os.path import join
from torch_geometric.data.data import BaseData

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.utils import load_pickle_form_pyg_data, loadBertModel
from tokenize_code import tokenize_code_line
from MultiFCode.embedding.vocabulary import Vocabulary

import psutil
import concurrent.futures

memory_base_path = ''

def generate_memory_base_path():
    global memory_base_path
    if memory_base_path == '':
        beijing_tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(beijing_tz).strftime('%Y-%m-%d_%H-%M-%S')
        memory_base_path = f'./model/internal_memory_{current_time}'
    if not os.path.exists(memory_base_path):
        os.makedirs(memory_base_path)
        print('---------------------------------------------------')
        print('memory_base_path created at:' + memory_base_path)
        print('---------------------------------------------------')


def memory_usage_psutil(str):
    global memory_base_path

    memory_path = os.path.join(memory_base_path, f'internal_memory.log')

    with open(memory_path, 'a') as file:
        mem = psutil.virtual_memory()
        file.write(f'{str}-- available: {mem.available}')
        file.write('\n')


class CodeDataSet(Dataset):
    def __init__(self, data_path_list, configs, root="./data", reprocess=False, one_time_read=False, pre_embed=False):
        self.data_path_list = data_path_list
        self.configs = configs
        self.g_type = list(configs.g_type)
        self.w2v_path = configs.model.w2v.w2v_path
        self.num_workers = configs.num_workers
        self.current_path_sep = os.path.sep
        self.reprocess = reprocess
        self.one_time_read = one_time_read
        self.pre_embed = pre_embed
        self.tuple_bert_model_novar = None
        self.tuple_bert_model_usevar = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.processed_dir_project_path = ''

        for t in self.g_type:
            if len([path for path in self.data_path_list if
                    path.split(self.current_path_sep)[-1].split('_')[0] == t]) > 1:
                raise Exception(f'Error in CodeDataSet, {t} type data path more than one', flush=True)

        self.graphs_path_dict_list = {}
        self.graphs_dict_list = {}

        super(CodeDataSet, self).__init__(root=root)

    def _process(self):
        
        self.processed_dir_project_path = join(self.processed_dir,
                                               *self.data_path_list[0].split(self.current_path_sep)[-3:-1])
        current_dir_paths = [join(self.processed_dir, *data_path.split(self.current_path_sep)[-3:]) for data_path in
                             self.data_path_list]
        if self.pre_embed:
            self.processed_dir_project_path = join(self.processed_dir, 'pre_embed',
                                                   *self.data_path_list[0].split(self.current_path_sep)[-3:-1])
            current_dir_paths = [join(self.processed_dir, 'pre_embed', *data_path.split(self.current_path_sep)[-3:]) for
                                 data_path in self.data_path_list]
            
            
            
            

        count_path = 0
        for current_dir_path in current_dir_paths:
            if os.path.exists(current_dir_path):
                count_path += 1

        
        if count_path == len(current_dir_paths):
            if not self.reprocess:
                if self.one_time_read:
                    
                    self.graphs_dict_list = self.get_pyg_data_list_one_time()
                    print('Processed data exists, no need to process again! one time read!', flush=True)
                    return
                else:
                    
                    self.graphs_path_dict_list = self.get_pyg_data_list()
                    print('Processed data exists, no need to process again!', flush=True)
                    return
            else:
                for current_dir_path in current_dir_paths:
                    if os.path.exists(current_dir_path):
                        shutil.rmtree(current_dir_path)
                        print(f'Remove old processed data: {current_dir_path}', flush=True)
        elif count_path != 0:
            if not self.reprocess:
                print('Please delete the data for the corresponding training type', flush=True)
                raise Exception('Error in CodeDataSet, processed data path error')
            else:
                for current_dir_path in current_dir_paths:
                    if os.path.exists(current_dir_path):
                        shutil.rmtree(current_dir_path)
                        print(f'Remove old processed data: {current_dir_path}', flush=True)

        print('Start process...', flush=True)

        for current_dir_path in current_dir_paths:
            os.makedirs(current_dir_path, exist_ok=True)
        self.process()

        print('Done process!', flush=True)

    def process(self):
        self.edge_type_index = {'ast': 1, 'cfg': 2, 'ddg': 3, 'cdg': 4, 'other': 0}

        self.vocab = Vocabulary.build_from_w2v(self.w2v_path)

        self.lock = threading.Lock()

        if self.pre_embed:
            print('Start load bert model...', flush=True)
            (config_usevar, tokenizer_usevar, model_usevar, config_novar, tokenizer_novar, model_novar) = loadBertModel(
                self.configs)
            self.tuple_bert_model_novar = (config_novar, tokenizer_novar, model_novar.to(self.device))
            self.tuple_bert_model_usevar = (config_usevar, tokenizer_usevar, model_usevar.to(self.device))
            print('End load bert model!', flush=True)

        for t in self.g_type:
            data_file_path_list = []
            
            self.max_token_len = -1
            
            self.max_token_len_func = -1
            
            self.graphs_file_path_list = []

            
            for data_path in self.data_path_list:
                current_type = data_path.split(self.current_path_sep)[-1].split('_')[0]
                if current_type == t:
                    file_path_list_ = os.listdir(data_path)
                    data_file_path_list.extend([(data_path, file_path) for file_path in file_path_list_])

            if self.pre_embed:
                print(f'Start call load_preprocess_save_embed method --{t}', flush=True)
                error_list = []
                for path_tuple in data_file_path_list:
                    future = self.load_preprocess_save_embed(path_tuple, t)
                    error_list.append(future)
                
                flag = False
                for error in error_list:
                    result = error
                    if result is not None:
                        flag = True
                        print('-----Error in load preprocess and save data-----', flush=True)
                        print(result, flush=True)
                        print('-----Error in load preprocess and save data-----', flush=True)
                if flag:
                    exit(1)
                print(f'End call load_preprocess_save_embed method! --{t}', flush=True)

                del data_file_path_list
                gc.collect()
            else:
                
                print(f'Start call load_preprocess_save_token method --{t}', flush=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    error_list = []
                    for path_tuple in data_file_path_list:
                        future = executor.submit(self.load_preprocess_save_token, path_tuple, t)
                        error_list.append(future)
                    executor.shutdown()
                    
                    flag = False
                    for error in error_list:
                        result = error.result()
                        if result is not None:
                            flag = True
                            print('-----Error in load preprocess and save data-----', flush=True)
                            print(result, flush=True)
                            print('-----Error in load preprocess and save data-----', flush=True)
                    if flag:
                        exit(1)

                print(f'End call load_preprocess_save_token method! --{t}', flush=True)

                del data_file_path_list
                gc.collect()

                
                if not self.pre_embed:
                    print(f'Start call deal_token_to_id_base_mtl method --{t}', flush=True)
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        error_list = []
                        for g_path in self.graphs_file_path_list:
                            future = executor.submit(self.deal_token_to_id_base_mtl, g_path)
                            error_list.append(future)
                        executor.shutdown()
                        
                        flag = False
                        for error in error_list:
                            result = error.result()
                            if result is not None:
                                flag = True
                                print('-----Error in deal token to id-----', flush=True)
                                print(result, flush=True)
                                print('-----Error in deal token to id-----', flush=True)
                        if flag:
                            exit(1)

                    print(f'End call deal_token_to_id_base_mtl method! --{t}', flush=True)
                    del self.graphs_file_path_list
                    gc.collect()
        if self.pre_embed:
            del config_usevar, tokenizer_usevar, model_usevar, config_novar, tokenizer_novar, model_novar, self.tuple_bert_model_novar, self.tuple_bert_model_usevar
            torch.cuda.empty_cache()
        del self.lock

        if self.one_time_read:
            
            self.graphs_dict_list = self.get_pyg_data_list_one_time()
        else:
            
            self.graphs_path_dict_list = self.get_pyg_data_list()

    def get_pyg_data_list(self):
        graphs_path_dict_list_new = {}
        for t in self.g_type:
            graphs_path_dict_list_new[t] = []
            for data_path in self.data_path_list:
                end_dir_path = data_path.split(self.current_path_sep)[-1]
                if end_dir_path.split('_')[0] == t:
                    current_dir_path = join(self.processed_dir_project_path,
                                            end_dir_path)  
                    file_path_list_ = os.listdir(current_dir_path)
                    
                    file_path_list_ = sorted(file_path_list_, key=lambda x: x.split('.')[0].split('-')[1])
                    
                    full_file_path_list_ = [join(current_dir_path, file_path) for file_path in file_path_list_]
                    graphs_path_dict_list_new[t] = full_file_path_list_
                    break
        return graphs_path_dict_list_new

    def get_pyg_data_list_one_time(self):
        graphs_dict_list_new = {}
        for t in self.g_type:
            graphs_dict_list_new[t] = []
            for data_path in self.data_path_list:
                end_dir_path = data_path.split(self.current_path_sep)[-1]
                if end_dir_path.split('_')[0] == t:
                    current_dir_path = join(self.processed_dir_project_path, end_dir_path)
                    file_path_list_ = os.listdir(current_dir_path)
                    file_path_list_ = sorted(file_path_list_, key=lambda x: x.split('.')[0].split('-')[1])
                    full_file_path_list_ = [join(current_dir_path, file_path) for file_path in file_path_list_]
                    
                    graphs_dict_list_new[t] = [torch.load(file_path) for file_path in full_file_path_list_]
        return graphs_dict_list_new

    def deal_token_to_id_base_mtl(self, g_path):
        g = torch.load(g_path)
        node_ids = torch.full((len(g.x), self.max_token_len),
                              self.vocab.get_pad_id(),
                              dtype=torch.long)
        node_ids_func = torch.full((1, self.max_token_len_func),
                                   self.vocab.get_pad_id(),
                                   dtype=torch.long)
        
        for i in range(len(g.x)):
            ids = self.vocab.convert_tokens_to_ids(g.x[i])
            node_ids[i, 0:len(ids)] = torch.tensor(ids, dtype=torch.long)

        
        ids_func = self.vocab.convert_tokens_to_ids(g.func)
        node_ids_func[0, 0:len(ids_func)] = torch.tensor(ids_func, dtype=torch.long)

        g.x = node_ids
        g.func = node_ids_func
        torch.save(g, g_path)

    def load_preprocess_save_token(self, data_file_tuple, current_type):
        try:
            pyg_data = load_pickle_form_pyg_data(join(data_file_tuple[0], data_file_tuple[1]))
            pyg_data.y = pyg_data.stores[0]['graph_label']
            pyg_data.x = pyg_data.stores[0]['label']

            
            
            

            
            if not isinstance(pyg_data.edge_index, torch.Tensor):
                pyg_data.edge_index = torch.tensor(pyg_data.edge_index, dtype=torch.long)

            pyg_data.graph_path = data_file_tuple[1]

            
            self.preproccess_edge_type(pyg_data, current_type)

            
            mtl, pyg_data.x = self.record_max_token_len_and_segmentation(pyg_data.x)
            mtl_func, pyg_data.func = self.record_max_token_len_and_segmentation_func(pyg_data.func)

            
            pyg_data.stores[0].pop('graph_label')
            pyg_data.stores[0].pop('label')
            if 'edge_label' in pyg_data.stores[0]:
                pyg_data.stores[0].pop('edge_label')

            
            current_processed_dir_path = join(self.processed_dir, *data_file_tuple[0].split(self.current_path_sep)[-3:])

            
            if not os.path.exists(current_processed_dir_path):
                os.makedirs(current_processed_dir_path, exist_ok=True)
            save_path = join(current_processed_dir_path, data_file_tuple[1])
            torch.save(pyg_data, join(save_path))

            
            with self.lock:
                if mtl > self.max_token_len:
                    self.max_token_len = mtl
                if mtl_func > self.max_token_len_func:
                    self.max_token_len_func = mtl_func
                self.graphs_file_path_list.append(save_path)

        except Exception as e:
            current_json = {
                'data_file_tuple': data_file_tuple,
                'current_type': current_type,
                'error': str(e)
            }
            return json.dumps(current_json)

    def load_preprocess_save_embed(self, data_file_tuple, current_type):
        try:
            config_novar, tokenizer_novar, model_novar = self.tuple_bert_model_novar
            config_usevar, tokenizer_usevar, model_usevar = self.tuple_bert_model_usevar
            pyg_data = load_pickle_form_pyg_data(join(data_file_tuple[0], data_file_tuple[1]))
            pyg_data.y = pyg_data.stores[0]['graph_label']
            pyg_data.x = pyg_data.stores[0]['label']

            
            
            

            
            if not isinstance(pyg_data.edge_index, torch.Tensor):
                pyg_data.edge_index = torch.tensor(pyg_data.edge_index, dtype=torch.long)

            pyg_data.graph_path = data_file_tuple[1]

            
            self.preproccess_edge_type(pyg_data, current_type)
            with torch.no_grad():
                max_len_x = 300
                
                for i in range(0, len(pyg_data.x), max_len_x):
                    inputs = tokenizer_usevar(pyg_data.x[i:i + max_len_x], padding=True, truncation=True,
                                              return_tensors="pt").to(self.device)
                    outputs = model_usevar(**inputs, output_hidden_states=True)
                    sentence_embeddings = outputs.last_hidden_state[:, 0]
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                    sentence_embeddings = sentence_embeddings.to('cpu')
                    pyg_data.x[i:i + max_len_x] = sentence_embeddings
                    del inputs, outputs, sentence_embeddings
                    torch.cuda.empty_cache()
                pyg_data.x = torch.stack(pyg_data.x, 0)

                inputs_func = tokenizer_novar(pyg_data.func, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs_func = model_novar(**inputs_func)
                sentence_embeddings_func = outputs_func.last_hidden_state[:, 0]
                sentence_embeddings_func = torch.nn.functional.normalize(sentence_embeddings_func, p=2, dim=1)
                sentence_embeddings_func = sentence_embeddings_func.to('cpu')
                pyg_data.func = sentence_embeddings_func

                del inputs_func, outputs_func, sentence_embeddings_func
                torch.cuda.empty_cache()

            
            pyg_data.stores[0].pop('graph_label')
            pyg_data.stores[0].pop('label')
            if 'edge_label' in pyg_data.stores[0]:
                pyg_data.stores[0].pop('edge_label')

            
            current_processed_dir_path = join(self.processed_dir, 'pre_embed',
                                              *data_file_tuple[0].split(self.current_path_sep)[-3:])

            
            if not os.path.exists(current_processed_dir_path):
                os.makedirs(current_processed_dir_path, exist_ok=True)
            save_path = join(current_processed_dir_path, data_file_tuple[1])
            torch.save(pyg_data, join(save_path))

        except Exception as e:
            current_json = {
                'data_file_tuple': data_file_tuple,
                'current_type': current_type,
                'error': str(e)
            }
            return json.dumps(current_json)

    def preproccess_edge_type(self, data, current_type):
        
        if 'edge_label' in data.stores[0]:
            edge_label = data.stores[0]['edge_label']
            edge_type = []
            for i in range(len(edge_label)):
                if 'DDG' in edge_label[i]:
                    edge_type.append(self.edge_type_index['ddg'])
                elif 'CDG' in edge_label[i]:
                    edge_type.append(self.edge_type_index['cdg'])
                else:
                    print(f'edge label error: {edge_label[i]}, index:{i}', flush=True)
                    edge_type.append(self.edge_type_index['other'])
            data.edge_type = edge_type
        else:
            if current_type == 'ast':
                data.edge_type = [self.edge_type_index['ast']] * len(data.edge_index[0])
            elif current_type == 'cfg':
                data.edge_type = [self.edge_type_index['cfg']] * len(data.edge_index[0])
            else:
                print(f'edge label error: {current_type}', flush=True)
                data.edge_type = [self.edge_type_index['other']] * len(data.edge_index[0])
        
        data.edge_type = torch.tensor(data.edge_type, dtype=torch.int64)

    def record_max_token_len_and_segmentation(self, X):
        mtl = -1
        for i in range(len(X)):
            X[i] = tokenize_code_line(X[i])
            if len(X[i]) > mtl:
                mtl = len(X[i])
        return mtl, X

    def record_max_token_len_and_segmentation_func(self, func):
        func_token = tokenize_code_line(func)
        mtl = len(func)
        return mtl, func_token

    def get(self, idx: int) -> BaseData:
        
        return_dict = {}
        if self.one_time_read:
            for t in self.g_type:
                return_dict[t] = self.graphs_dict_list[t][idx]
            return return_dict
        else:
            for t in self.g_type:
                return_dict[t] = torch.load(self.graphs_path_dict_list[t][idx])
            return return_dict

    def len(self) -> int:
        if self.one_time_read:
            return len(self.graphs_dict_list[self.g_type[0]]) if len(self.g_type) > 0 else 0
        else:
            return len(self.graphs_path_dict_list[self.g_type[0]]) if len(self.g_type) > 0 else 0


if __name__ == '__main__':
    dataset = CodeDataSet("../joern/data/pdg_output_pickle_test", "./data/word2vec/myw2v3.wv")
    print('Load done', flush=True)
    t = dataset.get(26)
    print(t)
    loader = DataLoader(dataset, 8, shuffle=False)
    for batch in loader:
        print(batch)
