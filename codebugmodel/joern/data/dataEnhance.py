import argparse
import copy
import json
import os
import pickle
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.utils import load_pickle_form_pyg_data, save_pyg_data_to_pickle

train_types = ['train']


def up_sample_by_cwe_project(project_name, sub_project_name, pickle_type):
    for train_type in train_types:
        for p_type in pickle_type:
            folder_path = f'./{project_name}/{sub_project_name}/{p_type}_output_pickle_{train_type}'
            if not os.path.exists(folder_path):
                continue

            files = os.listdir(folder_path)

            cwe_project_0_1_dict = {}
            cwe_project_id_up_num_dict = {}

            for file in files:
                file_path = os.path.join(folder_path, file)

                pyg_data = load_pickle_form_pyg_data(file_path)
                cwes = pyg_data.stores[0]['cwe']['cwe']
                project = pyg_data.stores[0]['project']
                target = pyg_data.y.item()
                for cwe in cwes:
                    key = cwe + '-%-' + project
                    cwe_project_0_1_dict.setdefault(key, []).append(target)

            for key, value in cwe_project_0_1_dict.items():
                count_0 = value.count(0)
                count_1 = value.count(1)

                if count_0 == 0 or count_1 == 0:
                    cwe_project_id_up_num_dict[key] = 0
                    continue
                add_multiple = count_0 // count_1
                cwe_project_id_up_num_dict[key] = add_multiple

            for file in files:
                idx = int(file.split('.')[0].split('-')[1])
                file_path = os.path.join(folder_path, file)

                pyg_data = load_pickle_form_pyg_data(file_path)
                cwes = pyg_data.stores[0]['cwe']['cwe']
                project = pyg_data.stores[0]['project']
                target = pyg_data.y.item()

                if target == 0:
                    pyg_data.stores[0]['idx_enhance'] = 'No'

                    save_pyg_data_to_pickle(pyg_data, file_path)
                    continue
                add_multiple = 0
                for cwe in cwes:
                    key = cwe + '-%-' + project
                    add_multiple += cwe_project_id_up_num_dict[key]

                avg_add_multiple = add_multiple // len(cwes)
                print(f'idx: {idx}, add_multiple: {add_multiple}, avg_add_multiple: {avg_add_multiple}')

                for i in range(1, avg_add_multiple):

                    new_g = copy.deepcopy(pyg_data)

                    func = new_g.stores[0]['func']
                    new_func = catch_func_name_and_replace(func, f'_enhance{i}')
                    new_g.stores[0]['func'] = new_func if new_func is not None else func

                    for index, node_label in enumerate(new_g.x):
                        new_node_label = match_and_replace(node_label, i, reg=r'VAR\d+')
                        new_node_label = match_and_replace(new_node_label, i, reg=r'FUN\d+')

                        new_g.x[index] = new_node_label
                    new_idx = str(idx) + f'_{i}'
                    new_g.stores[0]['idx_enhance'] = new_idx
                    new_g.stores[0]['label'] = new_g.x

                    new_file_path = file.replace(str(idx), new_idx)
                    save_pyg_data_to_pickle(new_g, os.path.join(folder_path, new_file_path))
                    print(f'Add file: {new_file_path}')
                    del new_g
                pyg_data.stores[0]['idx_enhance'] = str(idx) + '_0'

                save_pyg_data_to_pickle(pyg_data, file_path)


def up_and_down_sample(project_name, sub_project_name, pickle_type):
    statisticInfo = {}

    for train_type in train_types:
        for p_type in pickle_type:

            down_sample_id_0 = []
            down_sample_id_1 = []

            up_sample_id = []

            folder_path = f'./{project_name}/{sub_project_name}/{p_type}_output_pickle_{train_type}'
            if not os.path.exists(folder_path):
                continue

            files = os.listdir(folder_path)

            count_0 = 0
            count_1 = 0

            for file in files:
                file_path = os.path.join(folder_path, file)

                pyg_data = load_pickle_form_pyg_data(file_path)
                cwe = pyg_data.stores[0]['cwe']['cwe']
                graph_label = pyg_data.stores[0]['graph_label'].item()

                idx = pyg_data.stores[0]['idx'].item()
                if graph_label == 0:
                    count_0 += 1
                    if len(cwe) == 0:
                        down_sample_id_0.append(idx)

                        os.remove(file_path)
                        print(f'Delete file: {file_path}')
                elif graph_label == 1:
                    count_1 += 1
                    if len(cwe) == 0:
                        down_sample_id_1.append(idx)

                        os.remove(file_path)
                        print(f'Delete file: {file_path}')
                    else:

                        up_sample_id.append(idx)

            after_down_sample_count_0 = count_0 - len(down_sample_id_0)
            after_down_sample_count_1 = count_1 - len(down_sample_id_1)

            after_up_sample_count_1 = after_down_sample_count_1
            after_up_sample_count_0 = after_down_sample_count_0

            add_multiple = after_down_sample_count_0 // after_down_sample_count_1

            if add_multiple > 1:
                for file in files:
                    idx = int(file.split('.')[0].split('-')[1])
                    file_path = os.path.join(folder_path, file)

                    pyg_data = load_pickle_form_pyg_data(file_path)

                    if idx not in up_sample_id:
                        pyg_data.stores[0]['idx_enhance'] = 'No'

                        save_pyg_data_to_pickle(pyg_data, file_path)
                        continue

                    for i in range(1, add_multiple):

                        new_g = copy.deepcopy(pyg_data)

                        func = new_g.stores[0]['func']
                        new_func = catch_func_name_and_replace(func, f'_enhance{i}')
                        new_g.stores[0]['func'] = new_func if new_func is not None else func

                        for index, node_label in enumerate(new_g.x):
                            new_node_label = match_and_replace(node_label, i, reg=r'VAR\d+')
                            new_node_label = match_and_replace(new_node_label, i, reg=r'FUN\d+')

                            new_g.x[index] = new_node_label
                        new_idx = str(idx) + f'_{i}'
                        new_g.stores[0]['idx_enhance'] = new_idx
                        new_g.stores[0]['label'] = new_g.x

                        new_file_path = file.replace(str(idx), new_idx)
                        save_pyg_data_to_pickle(new_g, os.path.join(folder_path, new_file_path))
                        print(f'Add file: {new_file_path}')
                        after_up_sample_count_1 += 1
                        del new_g
                    pyg_data.stores[0]['idx_enhance'] = str(idx) + '_0'

                    save_pyg_data_to_pickle(pyg_data, file_path)

            statistic_info_circle = {
                f'{p_type}-{train_type}-down_sample': {
                    '0': after_down_sample_count_0,
                    '1': after_down_sample_count_1
                },
                f'{p_type}-{train_type}-up_sample': {
                    '0': after_up_sample_count_0,
                    '1': after_up_sample_count_1
                },
                f'{p_type}-{train_type}-add_multiple': add_multiple
            }

            statisticInfo.update(statistic_info_circle)
    print("----------------------------------------")
    print(statisticInfo)
    with open(f'./{project_name}/dataEnhance-statisticInfo.json', 'w') as statisticInfoFile:
        json.dump(statisticInfo, statisticInfoFile, indent=4)
    print("----------------------------------------")


def match_and_replace(node_label, index, reg=r'VAR\d+'):
    matches = re.finditer(reg, node_label)

    unique_matches = set()

    for match in matches:
        unique_matches.add(match.group())

    for match in unique_matches:
        match_str = match

        new_str = match_str + '_enhance' + str(index)
        node_label = node_label.replace(match_str, new_str)

    return node_label


def catch_func_name_and_replace(func, replace_content):
    reg = r'\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()'

    match = re.search(reg, func.split('\n')[0])

    if match is None:
        return None

    match_str = match.group()

    new_func_name = match_str + replace_content
    new_func = func.replace(match_str, new_func_name)
    return new_func


if __name__ == '__main__':

    project_name = 'DiverseVul-cwe10-simplify'
    sub_project_name = 'all'

    pickle_type = ['ast', 'cfg', 'pdg']

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--project_names', help='project_names')
    parser.add_argument('-sub', '--sub_project_names', help='sub_project_names')
    parser.add_argument('-t', '--pickle_type', nargs='+', help='pickle_type')

    args = parser.parse_args()

    if args.project_names:
        project_name = args.project_names
    if args.sub_project_names:
        sub_project_name = args.sub_project_names
    if args.pickle_type:
        pickle_type = args.pickle_type

    print('Start', flush=True)
    up_sample_by_cwe_project(project_name, sub_project_name, pickle_type)
    print('Done', flush=True)
