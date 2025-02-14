import argparse
import json
import os


def statistics_by_project_and_subproject(project_name, sub_project_name):
    statistics_path = os.path.join(project_name, f'statisticInfo_{sub_project_name}.json')
    root_path = os.path.join(project_name, sub_project_name)

    file_middle = '_output_pickle_'
    data_types = ['ast', 'cfg', 'pdg']
    train_types = ['train', 'test', 'valid']

    statistics = {}

    for train_type in train_types:
        for data_type in data_types:
            folder_path = os.path.join(root_path, f'{data_type}{file_middle}{train_type}')

            if not os.path.exists(folder_path):
                continue

            print(folder_path)

            
            files = os.listdir(folder_path)

            count_all = 0
            count_0 = 0
            count_1 = 0

            for file in files:
                file_type = int(file.split('.')[0].split('-')[-1])
                count_all += 1
                if file_type == 0:
                    count_0 += 1
                elif file_type == 1:
                    count_1 += 1
                else:
                    print(f'Error: {file}')

            statistics[f'{data_type}_{train_type}'] = {
                'count_all': count_all,
                'count_0': count_0,
                'count_1': count_1
            }

    with open(statistics_path, 'w') as f:
        json.dump(statistics, f, indent=4)



if __name__ == '__main__':
    projects = ['devign5']
    sub_projects = ['qemu']

    
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
