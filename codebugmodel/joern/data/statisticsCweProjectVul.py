import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.utils import load_pickle_form_pyg_data


def statistics_by_project_and_subproject(project_name, sub_project_name):
    statistics_path = os.path.join(project_name, f'statisticInfo_{sub_project_name}_cwe_project_vul_num.json')
    root_path = os.path.join(project_name, sub_project_name)

    file_middle = '_output_pickle_'
    data_types = ['ast', 'cfg', 'pdg']
    train_types = ['train', 'test', 'valid']

    statistics_num = {}

    for train_type in train_types:
        for data_type in data_types:
            folder_path = os.path.join(root_path, f'{data_type}{file_middle}{train_type}')

            if not os.path.exists(folder_path):
                continue

            print(folder_path)

            files = os.listdir(folder_path)

            current_statistics_num, current_statistics_count = statistics_cwe_project_vul_num(folder_path, files)
            statistics_num[f'{data_type}_{train_type}'] = current_statistics_num
            statistics_num[f'{data_type}_{train_type}_count'] = current_statistics_count

    with open(statistics_path, 'w') as f:
        json.dump(statistics_num, f, indent=4)


def statistics_cwe_project_vul_num(root_path, files):
    statistics_cwe_project_vul_num_dict = {}
    cwe_project_vul_num_dict = {}

    count_all = 0
    count_0 = 0
    count_1 = 0
    for file in files:
        file_path = os.path.join(root_path, file)

        pyg_data = load_pickle_form_pyg_data(file_path)
        cwes = pyg_data.stores[0]['cwe']['cwe']
        project = pyg_data.stores[0]['project']
        target = pyg_data.stores[0]['graph_label'].item()
        for cwe in cwes:
            key = cwe + '-%-' + project
            cwe_project_vul_num_dict.setdefault(key, []).append(target)
        count_all += 1
        if target == 0:
            count_0 += 1
        elif target == 1:
            count_1 += 1
        else:
            print(f'Error: {file}')
    statistics_count = {
        'count_all': count_all,
        'count_0': count_0,
        'count_1': count_1
    }
    for key in cwe_project_vul_num_dict:
        newkey_0 = key + '-%-' + '0'
        newkey_1 = key + '-%-' + '1'
        statistics_cwe_project_vul_num_dict[newkey_0] = cwe_project_vul_num_dict[key].count(0)
        statistics_cwe_project_vul_num_dict[newkey_1] = cwe_project_vul_num_dict[key].count(1)

    statistics_cwe_project_vul_num_dict = dict(
        sorted(statistics_cwe_project_vul_num_dict.items(), key=lambda item: item[0]))
    return statistics_cwe_project_vul_num_dict, statistics_count


if __name__ == '__main__':
    projects = ['DiverseVul-cwe10-simplify-enhance']
    sub_projects = ['all']

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--project_names', nargs='+', help='project_names')
    parser.add_argument('-sub', '--sub_project_names', nargs='+', help='sub_project_names')

    args = parser.parse_args()

    if args.project_names:
        projects = args.project_names
    if args.sub_project_names:
        sub_projects = args.sub_project_names

    print('Start')
    for project in projects:
        for sub_project in sub_projects:
            statistics_by_project_and_subproject(project, sub_project)
    print('Done')
