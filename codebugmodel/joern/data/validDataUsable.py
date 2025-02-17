import argparse
import json
import os
import pickle
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.utils import load_pickle_form_pyg_data

import concurrent.futures

my_lock = threading.Lock()
error_id_target_s = set()


def try_some_property(file_path):
    file_id = file_path.split(os.path.sep)[-1].split('.')[0].split('-')[1]

    pyg_data = load_pickle_form_pyg_data(file_path)
    try:
        if str(pyg_data.stores[0]['idx'].item()) != file_id:
            raise Exception('idx is not equal')
        if pyg_data.stores[0]['label'] is None or pyg_data.stores[0]['graph_label'] is None:
            raise Exception('x or y is None')
        if pyg_data.stores[0]['func'] is None:
            raise Exception('func is None')
        if pyg_data.stores[0]['idx'] is None:
            raise Exception('idx is None')
        if pyg_data.stores[0]['name'] is None:
            raise Exception('name is None')
        if 'project' in pyg_data.stores[0]:
            if pyg_data.stores[0]['project'] is None:
                raise Exception('project is None')
        if 'cwe' in pyg_data.stores[0]:
            if pyg_data.stores[0]['cwe']['cwe'] is None:
                raise Exception('cwe is None')
        num_nodes = pyg_data.num_nodes
        if num_nodes is None or num_nodes == 0:
            raise Exception('num_nodes is None')
    except Exception as e:
        current_json = {
            'Error path': file_path,
            'Message': str(e)
        }
        idx_target = '-'.join(file_path.split(os.path.sep)[-1].split('.')[0].split('-')[1:])

        with my_lock:
            error_id_target_s.add(idx_target)

        return json.dumps(current_json)


def valid_data_whether_usable(root_path):
    file_middle = '_output_pickle_'
    data_types = ['ast', 'cfg', 'pdg']
    train_types = ['train', 'test', 'valid']

    for train_type in train_types:

        error_id_target_s.clear()

        for data_type in data_types:
            folder_path = os.path.join(root_path, f'{data_type}{file_middle}{train_type}')

            if not os.path.exists(folder_path):
                continue

            print('Current:', folder_path, flush=True)

            files = os.listdir(folder_path)

            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                error_list = []
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    future = executor.submit(try_some_property, file_path)
                    error_list.append(future)
                executor.shutdown()
                for error in error_list:
                    result = error.result()
                    if result is not None:
                        print('--------------------------------------------------------------------------------',
                              flush=True)
                        print('-----Error in load preprocess and save data-----', flush=True)
                        print(result, flush=True)
                        print('-----Error in load preprocess and save data-----', flush=True)
                        print('--------------------------------------------------------------------------------',
                              flush=True)

        for error_id_target in error_id_target_s:
            for data_type in data_types:
                folder_path = os.path.join(root_path, f'{data_type}{file_middle}{train_type}')

                if not os.path.exists(folder_path):
                    continue

                file_path = os.path.join(folder_path, f'{data_type}-{error_id_target}.pkl')
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print('--------------------------------------------------------------------------------',
                          flush=True)
                    print(f'Delete file: {file_path}', flush=True)
                    print('--------------------------------------------------------------------------------',
                          flush=True)
                else:
                    print('--------------------------------------------------------------------------------',
                          flush=True)
                    print(f'File not exists: {file_path}', flush=True)
                    print('Error id:', error_id_target, flush=True)
                    print('Something wrong!', flush=True)
                    print('--------------------------------------------------------------------------------',
                          flush=True)


def valid_data_alignment(root_path):
    file_middle = '_output_pickle_'
    data_types = ['ast', 'cfg', 'pdg']
    train_types = ['train', 'test', 'valid']

    for train_type in train_types:

        error_id_target_s.clear()
        current_id_target_s = {}

        for data_type in data_types:
            folder_path = os.path.join(root_path, f'{data_type}{file_middle}{train_type}')

            if not os.path.exists(folder_path):
                continue

            print('Current:', folder_path, flush=True)

            files = os.listdir(folder_path)

            for file in files:
                id_target = '-'.join(file.split('.')[0].split('-')[1:])
                current_id_target_s.setdefault(id_target, []).append(data_type)

        for id_target, data_type_s in current_id_target_s.items():
            if len(data_type_s) != len(data_types):
                error_id_target_s.add(id_target)

        for error_id_target in error_id_target_s:
            for data_type in data_types:
                folder_path = os.path.join(root_path, f'{data_type}{file_middle}{train_type}')

                if not os.path.exists(folder_path):
                    continue

                file_path = os.path.join(folder_path, f'{data_type}-{error_id_target}.pkl')
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print('--------------------------------------------------------------------------------',
                          flush=True)
                    print(f'Delete file: {file_path}', flush=True)
                    print('--------------------------------------------------------------------------------',
                          flush=True)
                else:
                    print('--------------------------------------------------------------------------------',
                          flush=True)
                    print(f'File not exists: {file_path}', flush=True)
                    print('Error id:', error_id_target, flush=True)
                    print('Something wrong!', flush=True)
                    print('--------------------------------------------------------------------------------',
                          flush=True)


if __name__ == '__main__':
    projects = ['DiverseVul20k-simplify-test']
    sub_projects = ['all']

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--project_names', nargs='+', help='project_names')
    parser.add_argument('-sub', '--sub_project_names', nargs='+', help='sub_project_names')

    args = parser.parse_args()

    if args.project_names:
        projects = args.project_names
    if args.sub_project_names:
        sub_projects = args.sub_project_names

    print('Begin to check data whether usable', flush=True)

    for project in projects:
        for sub_project in sub_projects:
            root_path = os.path.join(project, sub_project)
            print('valid data whether usable:', root_path, flush=True)
            valid_data_whether_usable(root_path)
            print('valid data alignment:', root_path, flush=True)
            valid_data_alignment(root_path)

    print('End to check data whether usable', flush=True)
