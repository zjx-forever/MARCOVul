import argparse
import concurrent.futures
import os
import pickle
import re
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.utils import load_pickle_form_pyg_data, save_pyg_data_to_pickle
import numpy as np

NUM_WORKERS = 1


def simplifyMiddleStructureCode(data):
    data_X = data.x
    data_EdgeIndex = data.edge_index

    reg = r'<SUB>\d*?</SUB>'
    SUB_dict = {}

    data_X_ori = data_X.copy()

    for i in range(len(data_X)):
        current_SUB = re.search(reg, data_X[i])
        if current_SUB is None:
            continue

        data_X[i] = re.sub(reg, '', data_X[i])
        num = current_SUB.group()[5:-6]

        SUB_dict.setdefault(num, []).append(i)

    if len(SUB_dict) == 1:
        data.x = data_X_ori
        data.stores[0]['label'] = data_X_ori

        data.num_nodes = len(data_X_ori)
        data.num_edges = len(data.stores[0]['edge_index'][0])
        data.stores[0]['num_nodes'] = len(data_X_ori)
        data.stores[0]['num_edges'] = len(data.stores[0]['edge_index'][0])
        return data, 'skipped'

    data_X_need_delete = []

    for key in SUB_dict:
        key_0_0_index = SUB_dict[key][0]

        if len(SUB_dict[key]) <= 1:
            data_X[key_0_0_index] += ' <SUB>' + key + '</SUB>'
            continue

        for i in range(1, len(SUB_dict[key])):
            key_i_0_index = SUB_dict[key][i]

            data_X[key_0_0_index] += ' <SEP> ' + data_X[key_i_0_index]

            data_X_need_delete.append(key_i_0_index)

            for j in range(len(data_EdgeIndex[0])):
                if data_EdgeIndex[0][j] == key_i_0_index:
                    data_EdgeIndex[0][j] = key_0_0_index
                if data_EdgeIndex[1][j] == key_i_0_index:
                    data_EdgeIndex[1][j] = key_0_0_index

        data_X[key_0_0_index] += ' <SUB>' + key + '</SUB>'

    data_X_need_delete_set = set(data_X_need_delete)

    no_delete_data_X_index_dict = {}
    delete_count = 0
    new_data_X = []
    for i in range(len(data_X)):
        if i not in data_X_need_delete_set:
            no_delete_data_X_index_dict[i] = i - delete_count
            new_data_X.append(data_X[i])
        else:
            delete_count += 1

    data_X = new_data_X

    new_data_edge_index = [[], []]
    old_edge_label = []
    new_edge_label = []
    if 'edge_label' in data.stores[0]:
        old_edge_label = data.stores[0]['edge_label']

    for i in range(len(data_EdgeIndex[0])):
        data_EdgeIndex_0_i = data_EdgeIndex[0][i].item()
        data_EdgeIndex_1_i = data_EdgeIndex[1][i].item()

        if data_EdgeIndex_0_i != data_EdgeIndex_1_i:
            if data_EdgeIndex_0_i not in no_delete_data_X_index_dict or data_EdgeIndex_1_i not in no_delete_data_X_index_dict:
                raise Exception(
                    f'error in preproccess_merge_same_SUB, data_EdgeIndex[0][i]: {data_EdgeIndex_0_i}, data_EdgeIndex[1][i]: {data_EdgeIndex_1_i}')
            new_data_edge_index[0].append(no_delete_data_X_index_dict[data_EdgeIndex_0_i])
            new_data_edge_index[1].append(no_delete_data_X_index_dict[data_EdgeIndex_1_i])
            if 'edge_label' in data.stores[0]:
                new_edge_label.append(old_edge_label[i])

    new_data_edge_index = torch.tensor(new_data_edge_index, dtype=torch.long)
    data.edge_index = new_data_edge_index
    data.x = data_X
    data.stores[0]['label'] = data_X
    data.stores[0]['graph_label'] = data.y
    data.stores[0]['edge_index'] = new_data_edge_index
    if 'edge_label' in data.stores[0]:
        data.stores[0]['edge_label'] = new_edge_label
        data.edge_label = new_edge_label

    data.num_nodes = len(data_X)
    data.num_edges = len(new_data_edge_index[0])
    data.stores[0]['num_nodes'] = len(data_X)
    data.stores[0]['num_edges'] = len(new_data_edge_index[0])

    if np.max(np.array(data.edge_index)) >= len(data_X):
        raise Exception(
            f'error in preproccess_merge_same_SUB, max(data_EdgeIndex): {np.max(np.array(data.edge_index))}. Index beyond the boundary')

    return data, 'processed'


def load_pyg_data_preprocess(data_path):
    try:
        pyg_data = load_pickle_form_pyg_data(data_path)
        pyg_data.y = pyg_data.stores[0]['graph_label']
        pyg_data.x = pyg_data.stores[0]['label']

        pyg_data, status = simplifyMiddleStructureCode(pyg_data)

        data_path_split = data_path.split(os.path.sep)
        data_path_split[-4] = f'{data_path_split[-4]}-simplify'
        new_pyg_data_path = os.path.sep.join(data_path_split)
        if not os.path.exists(os.path.sep.join(data_path_split[:-1])):
            os.makedirs(os.path.sep.join(data_path_split[:-1]), exist_ok=True)

        save_pyg_data_to_pickle(pyg_data, new_pyg_data_path)

        return status
    except Exception as e:
        return f'Error in simplify middle structure code: {data_path}, {str(e)}'


def multiprocess_data(root_path):
    data_file_path_list = []
    for root, dirs, files in os.walk(root_path):
        parent_dir = os.path.basename(root)
        if '_output_pickle_' not in parent_dir:
            continue
        for file in files:
            if file.endswith('.pkl'):
                data_file_path_list.append(os.path.join(root, file))

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for data_file_path in data_file_path_list:
            future = executor.submit(load_pyg_data_preprocess, data_file_path)
            futures.append(future)
        executor.shutdown()

        skipped_count = 0
        processed_count = 0
        error_count = 0

        flag = False
        for future in futures:
            result = future.result()
            if result is not None:
                if result.startswith('Error'):
                    flag = True
                    error_count += 1
                    print('-----Error in simplify middle structure code-----', flush=True)
                    print(result, flush=True)
                    print('-----Error in simplify middle structure code-----', flush=True)
                elif result == 'skipped':
                    skipped_count += 1
                elif result == 'processed':
                    processed_count += 1

        if flag:
            exit(1)

        return skipped_count, processed_count, error_count


if __name__ == '__main__':

    projects = ['test']
    sub_projects = ['all']

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--project_names', nargs='+', help='project_names')
    parser.add_argument('-sub', '--sub_project_names', nargs='+', help='sub_project_names')
    parser.add_argument('-n', '--num_workers', type=int, help='num_workers')

    args = parser.parse_args()

    if args.project_names:
        projects = args.project_names
    if args.sub_project_names:
        sub_projects = args.sub_project_names
    if args.num_workers:
        NUM_WORKERS = int(args.num_workers)

    print('Start simplify middle structure code', flush=True)

    total_skipped = 0
    total_processed = 0
    total_errors = 0

    for project in projects:
        for sub_project in sub_projects:
            root_path = os.path.join(project, sub_project)
            print(f'Processing {root_path}...', flush=True)
            skipped, processed, errors = multiprocess_data(root_path)
            total_skipped += skipped
            total_processed += processed
            total_errors += errors
            print(f'  Skipped: {skipped}, Processed: {processed}, Errors: {errors}', flush=True)

    print('Done simplify middle structure code', flush=True)
    print('***** Processing Statistics *****', flush=True)
    print(f'  Total files skipped: {total_skipped}', flush=True)
    print(f'  Total files processed: {total_processed}', flush=True)
    print(f'  Total errors: {total_errors}', flush=True)
    print(f'  Total files: {total_skipped + total_processed + total_errors}', flush=True)
