import argparse
import copy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import random
import sys
import time
from typing import cast
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
import torch_geometric
from tqdm import tqdm
from datetime import datetime
import pytz
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef, \
    precision_recall_curve
import wandb

torch.multiprocessing.set_sharing_strategy('file_system')

from model import MulModel_GCN_GCN_RGCN_LLM, MulModel_Single_Test_LLM, Single_Text_LLM, MulModel_Four_modules_LLM
from dataSet import CodeDataSet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.parse_args import configure_arg_parser

MY_MODEL_CLASSES = {
    'GCN_GCN_RGCN_LLM': MulModel_GCN_GCN_RGCN_LLM,
    'MulModel_Single_Test_LLM': MulModel_Single_Test_LLM,
    'Single_Text_LLM': Single_Text_LLM,
    'MulModel_Four_modules_LLM': MulModel_Four_modules_LLM,
}

tqdm_disable = True


def keep_reproducibility(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch_geometric.seed_everything(seed)


myseed = 422

keep_reproducibility(myseed)

whether_shuffle = True
whether_pin_memory = False
whether_one_time_read = True
whether_reprocess = False
whether_persistent_workers = True

num_prefetch_factor_train = 2
num_prefetch_factor_valid = 2
num_prefetch_factor_test = 2

whether_log_wandb = False
whether_pre_embed = False

grad_accumulation_steps = 1
min_learning_rate = None

config = None
best_model_file_path = ''
current_time = ''
model_base_path = ''


def train_pre_train(model_type, device):
    global myseed
    global config
    global best_model_file_path, whether_reprocess
    model = None
    ast_best_model_file_path = ''
    cfg_best_model_file_path = ''
    pdg_best_model_file_path = ''

    if whether_reprocess:

        ori_config_gtype = copy.deepcopy(config.g_type)
        config.g_type = ['ast', 'cfg', 'pdg']

        train_path_list = generate_path_list('train')
        valid_path_list = generate_path_list('valid')
        test_path_list = generate_path_list('test')

        if os.path.sep == '\\':

            for i in range(len(train_path_list)):
                train_path_list[i] = os.path.normpath(train_path_list[i])
            for i in range(len(valid_path_list)):
                valid_path_list[i] = os.path.normpath(valid_path_list[i])

        print(f'------------------------Init------------------------', flush=True)
        print('Init Train Data...', flush=True)
        CodeDataSet(train_path_list, config, reprocess=whether_reprocess,
                    one_time_read=whether_one_time_read, pre_embed=whether_pre_embed)
        print('Init Train Data Done...', flush=True)

        print('Init Valid Data...', flush=True)
        CodeDataSet(valid_path_list, config, reprocess=whether_reprocess,
                    one_time_read=whether_one_time_read, pre_embed=whether_pre_embed)
        print('Init Valid Data Done...', flush=True)

        print('Init Test Data...')
        CodeDataSet(test_path_list, config, reprocess=whether_reprocess,
                    one_time_read=whether_one_time_read, pre_embed=whether_pre_embed)
        print('Init Test Data Done...')
        print(f'----------------------------------------------------', flush=True)

        config.g_type = ori_config_gtype

    def train_by_type(type_list):
        keep_reproducibility(myseed)
        generate_base_info()
        save_model_file()
        rng_state()
        config.g_type = type_list
        model = MY_MODEL_CLASSES[model_type](config)
        train_valid(model, device)
        current_best_model_file_path = copy.deepcopy(best_model_file_path)
        test(model, device)
        del model
        torch.cuda.empty_cache()
        return current_best_model_file_path

    if config.pre_train_structure.exist:
        ast_best_model_file_path = config.pre_train_structure.ast_path
        cfg_best_model_file_path = config.pre_train_structure.cfg_path
        pdg_best_model_file_path = config.pre_train_structure.pdg_path
    else:
        ori_use_text = copy.deepcopy(config.use_text)
        ori_pre_train_structure_used = copy.deepcopy(config.pre_train_structure.used)
        ori_whether_reprocess = copy.deepcopy(whether_reprocess)

        config.use_text = False
        config.pre_train_structure.used = False
        whether_reprocess = False
        print('------------------------Train Begin------------------------', flush=True)
        print('------------------------Train AST Begin------------------------', flush=True)
        ast_best_model_file_path = train_by_type(['ast'])
        print('------------------------Train AST Done------------------------', flush=True)
        print('------------------------Train CFG Begin------------------------', flush=True)
        cfg_best_model_file_path = train_by_type(['cfg'])
        print('------------------------Train CFG Done------------------------', flush=True)
        print('------------------------Train PDG Begin------------------------', flush=True)
        pdg_best_model_file_path = train_by_type(['pdg'])
        print('------------------------Train PDG Done------------------------', flush=True)
        config.use_text = ori_use_text
        config.pre_train_structure.used = ori_pre_train_structure_used
        whether_reprocess = ori_whether_reprocess

    print('------------------------Train All Begin------------------------', flush=True)
    print('------------------------Best Model File Path------------------------', flush=True)
    print('AST: ', ast_best_model_file_path)
    print('CFG: ', cfg_best_model_file_path)
    print('PDG: ', pdg_best_model_file_path)
    print('------------------------Best Model File Path------------------------', flush=True)
    keep_reproducibility(myseed)
    generate_base_info()
    save_model_file()
    rng_state()
    config.g_type = ['ast', 'cfg', 'pdg']
    model = MY_MODEL_CLASSES[model_type](config)

    prefixes_to_exclude = ['classifierLayer']

    ast_pretrained_dict = torch.load(ast_best_model_file_path)
    ast_filtered_dict = {k: v for k, v in ast_pretrained_dict.items() if
                         not any(k.startswith(prefix) for prefix in prefixes_to_exclude)}
    ast_filtered_dict = {k.replace('MLPSequential', 'ast_MLP') if 'MLPSequential' in k else k: v for k, v in
                         ast_filtered_dict.items()}
    cfg_pretrained_dict = torch.load(cfg_best_model_file_path)
    cfg_filtered_dict = {k: v for k, v in cfg_pretrained_dict.items() if
                         not any(k.startswith(prefix) for prefix in prefixes_to_exclude)}
    cfg_filtered_dict = {k.replace('MLPSequential', 'cfg_MLP') if 'MLPSequential' in k else k: v for k, v in
                         cfg_filtered_dict.items()}
    pdg_pretrained_dict = torch.load(pdg_best_model_file_path)
    pdg_filtered_dict = {k: v for k, v in pdg_pretrained_dict.items() if
                         not any(k.startswith(prefix) for prefix in prefixes_to_exclude)}
    pdg_filtered_dict = {k.replace('MLPSequential', 'pdg_MLP') if 'MLPSequential' in k else k: v for k, v in
                         pdg_filtered_dict.items()}

    model.load_state_dict(ast_filtered_dict, strict=False)
    model.load_state_dict(cfg_filtered_dict, strict=False)
    model.load_state_dict(pdg_filtered_dict, strict=False)

    ori_whether_reprocess = copy.deepcopy(whether_reprocess)
    whether_reprocess = False

    train_valid(model, device)

    test(model, device)
    whether_reprocess = ori_whether_reprocess
    print('------------------------Train All Done------------------------', flush=True)
    print('------------------------Train Done------------------------', flush=True)
    return model


def train_valid(model, device):
    global config
    global best_model_file_path

    global whether_shuffle, whether_log_wandb, whether_one_time_read, whether_pin_memory, grad_accumulation_steps

    global myseed

    learning_rate_config = config.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_config.init)
    optimizer.zero_grad()
    scheduler = StepLR(optimizer, step_size=learning_rate_config.step_size, gamma=learning_rate_config.gamma)

    model = model.to(device)

    train_path_list = generate_path_list('train')
    valid_path_list = generate_path_list('valid')

    if os.path.sep == '\\':

        for i in range(len(train_path_list)):
            train_path_list[i] = os.path.normpath(train_path_list[i])
        for i in range(len(valid_path_list)):
            valid_path_list[i] = os.path.normpath(valid_path_list[i])

    print(f'------------------------Train------------------------', flush=True)
    print('Loading Train Data...', flush=True)
    dataset_train = CodeDataSet(train_path_list, config, reprocess=whether_reprocess,
                                one_time_read=whether_one_time_read, pre_embed=whether_pre_embed)
    print('Train Data Loading Done...', flush=True)
    loader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=whether_shuffle,
                              num_workers=config.num_workers, drop_last=False, pin_memory=whether_pin_memory,
                              prefetch_factor=num_prefetch_factor_train, persistent_workers=whether_persistent_workers)

    print('Loading Valid Data...', flush=True)
    dataset_valid = CodeDataSet(valid_path_list, config, reprocess=whether_reprocess,
                                one_time_read=whether_one_time_read, pre_embed=whether_pre_embed)
    print('Valid Data Loading Done...', flush=True)
    loader_valid = DataLoader(dataset_valid, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, drop_last=False, pin_memory=whether_pin_memory,
                              prefetch_factor=num_prefetch_factor_valid, persistent_workers=whether_persistent_workers)

    if device == 'cuda':
        torch.cuda.empty_cache()

    all_epoch = config.epoch
    early_stop_cnt = 0

    min_loss = 1000.
    max_all_score = 0.
    loss_record = {'train': [], 'valid': []}
    print('---------------------------------------------Start---------------------------------------------', flush=True)
    g_type = list(config.g_type)
    for epoch in range(all_epoch):
        model.train()
        all_preds = np.array([])
        all_labels = np.array([])

        with tqdm(total=len(loader_train), ncols=80, desc=f"Epoch [{epoch + 1}/{all_epoch}]", disable=tqdm_disable,
                  mininterval=300) as pbar:
            total_loss = 0

            for i, batch in enumerate(loader_train):
                for t in g_type:
                    batch[t].to(device)

                output = model(batch)
                class_weight = list(config.class_weight)
                loss = model.cal_loss(output, batch[g_type[0]].y, config, class_weights=class_weight)

                loss.backward()

                if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(loader_train):
                    optimizer.step()

                    optimizer.zero_grad()

                total_loss += loss.detach().cpu().item() * len(batch[g_type[0]])
                loss_record['train'].append(loss.detach().cpu().item())

                preds = output.argmax(dim=1)
                all_preds = np.append(all_preds, preds.cpu().numpy())
                all_labels = np.append(all_labels, batch[g_type[0]].y.cpu().numpy())

                pbar.update(1)

        if min_learning_rate is None or optimizer.param_groups[0]['lr'] > min_learning_rate:
            print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')
            scheduler.step()

        log_dict = {}

        train_acc, train_recall, train_precision, train_f1, train_macro_f1, train_mcc = calculate_metrics(all_preds,
                                                                                                          all_labels)

        avg_loss = total_loss / len(loader_train.dataset)
        print('Train:', flush=True)
        print(f'Train Epuch: {epoch}', flush=True)
        print(
            f'Train Acc: {train_acc}, Train Recall: {train_recall}, Train Precision: {train_precision}, Train F1: {train_f1},  Train Macro-F1: {train_macro_f1}, Train MCC: {train_mcc}',
            flush=True)
        print('---------------------------------', flush=True)

        log_dict["Epoch"] = epoch
        log_dict["train_acc"] = train_acc
        log_dict["train_recall"] = train_recall
        log_dict["train_precision"] = train_precision
        log_dict["train_f1"] = train_f1
        log_dict["train_macro_f1"] = train_macro_f1
        log_dict["train_mcc"] = train_mcc
        log_dict["train_Avgloss"] = avg_loss

        print('Valid Start:', flush=True)

        validAvgloss, valid_acc, valid_recall, valid_precision, valid_f1, valid_macro_f1, valid_mcc = valid(model,
                                                                                                            loader_valid,
                                                                                                            device)
        print('Valid End!', flush=True)

        log_dict["valid_acc"] = valid_acc
        log_dict["valid_recall"] = valid_recall
        log_dict["valid_precision"] = valid_precision
        log_dict["valid_f1"] = valid_f1
        log_dict["valid_macro_f1"] = valid_macro_f1
        log_dict["valid_mcc"] = valid_mcc
        log_dict["valid_Avgloss"] = validAvgloss

        all_score = valid_macro_f1

        if all_score > max_all_score:
            max_all_score = all_score

            early_stop_cnt = 0

            global model_base_path
            best_model_file_path = os.path.join(model_base_path,
                                                f'model_epoch_{epoch}_acc_{valid_acc}_f1_{valid_f1}_recall_{valid_recall}_precision_{valid_precision}.pth')
            torch.save(model.state_dict(), best_model_file_path)

            print('---------Current Best Model---------', flush=True)

            print('Current all_score: ', all_score, flush=True)
            print(
                f"Best Valid epoch: {epoch}, Best Valid Acc: {valid_acc}, Best Valid Recall: {valid_recall}, Best Valid Precision: {valid_precision}, Best Valid F1: {valid_f1}, Best Valid Macro-F1: {valid_macro_f1}, Best Valid MCC: {valid_mcc}",
                flush=True)
        else:
            early_stop_cnt += 1

        loss_record['valid'].append(validAvgloss)

        if early_stop_cnt >= config.early_stop:
            print('Early Stop!', flush=True)
            break

        if whether_log_wandb:
            wandb.log(log_dict)

        print('------------------------------------------------------------------------------------------', flush=True)
        print('\n')

    print('Best Model File Path: ', best_model_file_path)
    print('---------------------------------------------Done---------------------------------------------', flush=True)

    return min_loss, loss_record


def valid(model, valid_set, device):
    global config

    g_type = list(config.g_type)

    model.eval()

    with torch.no_grad():
        all_preds = np.array([])
        all_labels = np.array([])
        total_loss = 0
        for batch in valid_set:
            for t in g_type:
                batch[t].to(device)
            output = model(batch)
            class_weight = list(config.class_weight)
            loss = model.cal_loss(output, batch[g_type[0]].y, config, class_weights=class_weight)
            total_loss += loss.detach().cpu().item() * len(batch[g_type[0]])

            preds = output.argmax(dim=1)
            all_preds = np.append(all_preds, preds.cpu().numpy())
            all_labels = np.append(all_labels, batch[g_type[0]].y.cpu().numpy())

        avg_loss = total_loss / len(valid_set.dataset)
        valid_acc, valid_recall, valid_precision, valid_f1, valid_macro_f1, valid_mcc = calculate_metrics(all_preds,
                                                                                                          all_labels)
        print(
            f'Valid Acc: {valid_acc}, Valid Recall: {valid_recall}, Valid Precision: {valid_precision}, Valid F1: {valid_f1},  Valid Macro-F1: {valid_macro_f1}, Valid MCC: {valid_mcc}')

        return avg_loss, valid_acc, valid_recall, valid_precision, valid_f1, valid_macro_f1, valid_mcc


def test(model, device):
    global config
    global best_model_file_path
    global whether_log_wandb, whether_one_time_read, whether_pin_memory

    if best_model_file_path == '' or best_model_file_path is None:
        print("Error: best_model_file_path is '' or None")
        return

    test_path_list = generate_path_list('test')

    print('---------------------------------------------Test---------------------------------------------')
    print('model file path: ', best_model_file_path)

    model.load_state_dict(torch.load(best_model_file_path))
    model.eval()
    print('Loading Test Data...')
    dataset_test = CodeDataSet(test_path_list, config, reprocess=whether_reprocess, one_time_read=whether_one_time_read,
                               pre_embed=whether_pre_embed)
    print('Test Data Loading Done...')
    loader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, drop_last=False, pin_memory=whether_pin_memory,
                             prefetch_factor=num_prefetch_factor_test, persistent_workers=whether_persistent_workers)

    g_type = list(config.g_type)

    if device == 'cuda':
        torch.cuda.empty_cache()

    start_time = time.time()

    with torch.no_grad():
        all_preds = np.array([])
        all_labels = np.array([])
        all_ids = np.array([])

        with tqdm(total=len(loader_test), ncols=80, desc=f"Process:", disable=tqdm_disable, mininterval=300) as pbar:
            for batch in loader_test:
                for t in g_type:
                    batch[t].to(device)
                output = model(batch)
                preds = output.argmax(dim=1)
                all_preds = np.append(all_preds, preds.cpu().numpy())
                all_labels = np.append(all_labels, batch[g_type[0]].y.cpu().numpy())
                all_ids = np.append(all_ids, batch[g_type[0]].idx.cpu().numpy())

                pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_sample = total_time / len(all_labels)

    test_acc, test_recall, test_precision, test_f1, test_macro_f1, test_mcc, pr_dict = calculate_metrics(all_preds,
                                                                                                         all_labels,
                                                                                                         pr_curve=True)
    print(
        f'Test Acc: {test_acc}, Test Recall: {test_recall}, Test Precision: {test_precision}, Test F1: {test_f1},  Test Macro-F1: {test_macro_f1}, Test MCC: {test_mcc}')

    print("***** Test Time Statistics *****")
    print(f"  Total test time: {total_time:.4f} seconds")
    print(f"  Average time per sample: {avg_time_per_sample:.4f} seconds")
    print(f"  Samples per second: {1.0 / avg_time_per_sample:.4f}")

    if whether_log_wandb:
        log_dict_test = {
            "test_acc": test_acc,
            "test_recall": test_recall,
            "test_precision": test_precision,
            "test_f1": test_f1,
            "test_macro_f1": test_macro_f1,
            "test_mcc": test_mcc,
            "total_test_time": total_time,
            "avg_time_per_sample": avg_time_per_sample,
            "samples_per_second": 1.0 / avg_time_per_sample
        }
        wandb.log(log_dict_test)

    beijing_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(beijing_tz).strftime('%Y-%m-%d_%H-%M-%S')
    global model_base_path
    output_result(all_ids, all_preds, all_labels,
                  os.path.join(model_base_path, f'output_result_{current_time}.txt'))
    pr_curve_path = os.path.join(model_base_path, f'pr_curve_{current_time}.pth')
    torch.save(pr_dict, pr_curve_path)

    print('---------------------------------------------Done---------------------------------------------')


def calculate_metrics(all_preds, all_labels, pr_curve=False):
    test_acc = accuracy_score(all_labels, all_preds)

    test_recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)

    test_precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)

    test_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)

    test_macro_f1 = f1_score(all_labels, all_preds, average='macro')

    test_mcc = matthews_corrcoef(all_labels, all_preds)

    if pr_curve:
        precision_list, recall_list, thresholds_list = precision_recall_curve(all_labels, all_preds)
        pr_dict = {'precision': precision_list, 'recall': recall_list, 'thresholds': thresholds_list}
        return test_acc, test_recall, test_precision, test_f1, test_macro_f1, test_mcc, pr_dict
    return test_acc, test_recall, test_precision, test_f1, test_macro_f1, test_mcc


def output_result(ids, preds, labels, file_path):
    with open(file_path, 'w') as f:
        for i in range(len(labels)):
            f.write(f'{ids[i]}\t\tA:{labels[i]}\t\tP:{preds[i]}\n')


def rng_state():
    global current_time
    global model_base_path

    rng_state = torch.get_rng_state()
    rng_state_path = os.path.join(model_base_path, f'rng_state_{current_time}.pth')
    torch.save(rng_state, rng_state_path)
    print('---------------------------------------------------')
    print('rng_state saved at:' + rng_state_path)
    print('---------------------------------------------------')


def save_model_file():
    global model_base_path

    model_file_path = './model.py'

    os.system(f'cp {model_file_path} {model_base_path}')
    print('---------------------------------------------------')
    print(f'model file {model_file_path} copy to: {model_base_path}')
    print('---------------------------------------------------')

    config_file_path = '../configs/config.yaml'
    os.system(f'cp {config_file_path} {model_base_path}')
    print('---------------------------------------------------')
    print(f'config file {config_file_path} copy to: {model_base_path}')
    print('---------------------------------------------------')


def generate_base_info():
    global current_time
    global model_base_path

    beijing_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(beijing_tz).strftime('%Y-%m-%d_%H-%M-%S')

    model_base_path = f'./model/model_{current_time}'
    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)
    print('---------------------------------------------------')
    print('model_base_path created at:' + model_base_path)
    print('---------------------------------------------------')


def generate_path_list(pattern):
    global config

    path_list = []
    g_type = list(config.g_type)
    if pattern == 'train':
        if 'ast' in g_type:
            path_list.append(config.ast_train_data_path)
        if 'cfg' in g_type:
            path_list.append(config.cfg_train_data_path)
        if 'pdg' in g_type:
            path_list.append(config.pdg_train_data_path)






    elif pattern == 'valid':
        if 'ast' in g_type:
            path_list.append(config.ast_valid_data_path)
        if 'cfg' in g_type:
            path_list.append(config.cfg_valid_data_path)
        if 'pdg' in g_type:
            path_list.append(config.pdg_valid_data_path)
    elif pattern == 'test':
        if 'ast' in g_type:
            path_list.append(config.ast_test_data_path)
        if 'cfg' in g_type:
            path_list.append(config.cfg_test_data_path)
        if 'pdg' in g_type:
            path_list.append(config.pdg_test_data_path)

    return path_list


def main():
    global config
    global best_model_file_path
    global whether_log_wandb, whether_pre_embed, whether_reprocess

    arg_parser = configure_arg_parser()
    args, unknown = arg_parser.parse_known_args()
    config = cast(DictConfig, OmegaConf.load(args.config))
    print('-------------Config-------------')
    print(OmegaConf.to_yaml(config))
    print('--------------------------------')

    parser = argparse.ArgumentParser()

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('-dsn', '--data_set_name', help='data_set_name')
    parser.add_argument('-dssub', '--sub_project', help='sub_project')

    parser.add_argument('-w2vp', '--w2v_project_name', help='w2v_project_name')
    parser.add_argument('-w2vsub', '--w2v_sub_project_name', help='w2v_sub_project_name')
    parser.add_argument('-w2vsg', '--w2v_skip_gram', type=int, help='w2v_skip_gram')
    parser.add_argument('-model', '--use_model', help='use_model', required=True)

    parser.add_argument('-gt', '--gType', nargs='+', type=str, help='a list of gType')
    parser.add_argument('-ut', '--useText', type=str2bool, help='Set useText')
    parser.add_argument('-pe', '--preEmbed', type=str2bool, help='Set preEmbed')
    parser.add_argument('-ptsu', '--PTSUsed', type=str2bool, help='Set pre_train_structure.used')
    parser.add_argument('-ptse', '--PTSExist', type=str2bool, help='Set pre_train_structure.exist')

    parser.add_argument('-wr', '--whetherReprocess', type=str2bool, help='Set whether_reprocess')

    parser.add_argument('-gtmAST', '--g_type_model_ast', help='Set g_type_model_ast')
    parser.add_argument('-gtmCFG', '--g_type_model_cfg', help='Set g_type_model_cfg')
    parser.add_argument('-gtmPDG', '--g_type_model_pdg', help='Set g_type_model_pdg')

    args = parser.parse_args()

    if args.w2v_project_name:
        config.model.w2v.project_name = args.w2v_project_name
    if args.w2v_sub_project_name:
        config.model.w2v.sub_project = args.w2v_sub_project_name
    if args.w2v_skip_gram:
        config.model.w2v.sg = int(args.w2v_skip_gram)
        if config.model.w2v.sg == 1:
            config.model.w2v.sg_name = "Skip-gram"
        else:
            config.model.w2v.sg_name = "CBOW"

    if args.data_set_name:
        config.data_set_name = args.data_set_name
    if args.sub_project:
        config.sub_project = args.sub_project

    if args.gType:
        config.g_type = args.gType
    if args.useText is not None:
        config.use_text = args.useText
    if args.preEmbed is not None:
        config.pre_embed = args.preEmbed
    if args.PTSUsed is not None:
        config.pre_train_structure.used = args.PTSUsed
    if args.PTSExist is not None:
        config.pre_train_structure.exist = args.PTSExist

    if args.g_type_model_ast:
        config.g_type_model.ast = args.g_type_model_ast
    if args.g_type_model_cfg:
        config.g_type_model.cfg = args.g_type_model_cfg
    if args.g_type_model_pdg:
        config.g_type_model.pdg = args.g_type_model_pdg

    if args.whetherReprocess is not None:
        whether_reprocess = args.whetherReprocess

    whether_log_wandb = config.wandb_log
    whether_pre_embed = config.pre_embed

    print("current g_type: ", list(config.g_type))

    if whether_log_wandb:
        wandb.login()
        wandb.init(

            project="Bug-MultiFCode-zjx-Fix3",

            config={
                "seed": myseed,
                "batch_size": config.batch_size,
                "epoch": config.epoch,
                "L1_alpha": config.L1_alpha,
                "L2_alpha": config.L2_alpha,
                "learning_rate_init": config.learning_rate.init,
                "learning_rate_step_size": config.learning_rate.step_size,
                "learning_rate_gamma": config.learning_rate.gamma,
            }
        )

        wandb.define_metric("Epoch")
        wandb.define_metric("train_*", step_metric="Epoch")
        wandb.define_metric("valid_*", step_metric="Epoch")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f'device: {device}')

    best_model_file_path = ''

    if config.pre_train_structure.used:

        model = train_pre_train(args.use_model, device)
    else:
        generate_base_info()
        save_model_file()

        rng_state()

        model = MY_MODEL_CLASSES[args.use_model](config)

        if config.fine_tuning:
            model.load_state_dict(torch.load(config.best_model_file_path))
            print('Load model from: ', config.best_model_file_path)

        train_valid(model, device)

        test(model, device)

    if whether_log_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
