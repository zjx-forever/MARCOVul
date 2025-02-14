import argparse
import os

dataset_name = 'DiverseVul-valid'

sub_dataset_name = 'all'

middle_dir_name = '_output_pickle_'
gtype = ['ast', 'cfg', 'pdg']
train_type = ['train', 'test', 'valid']

def main():
    global dataset_name, sub_dataset_name

    usable_data = set()
    count_0 = 0
    count_1 = 0

    root_dataset_path = os.path.join(dataset_name, sub_dataset_name)
    print(root_dataset_path)
    for train_type_name in train_type:
        dir_name = gtype[0] + middle_dir_name + train_type_name
        current_dir_path = os.path.join(root_dataset_path, dir_name)
        if not os.path.exists(current_dir_path):
            print('current dir not exists:', current_dir_path)
            continue
        files = os.listdir(current_dir_path)
        for file in files:
            file_id = file.split('.')[0].split('-')[1]
            usable_data.add(file_id)
            vul = int(file.split('.')[0].split('-')[2])
            if vul == 0:
                count_0 += 1
            elif vul == 1:
                count_1 += 1
            else:
                print('error')
    print('count_0', count_0)
    print('count_1', count_1)
    with open(f'{dataset_name}-usable_data.txt', 'w') as f:
        for data in usable_data:
            f.write(data + '\n')

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()

    
    parser.add_argument('-p', '--project_name', help='project_name')
    parser.add_argument('-sub', '--sub_dataset_name', help='sub_dataset_name')

    
    args = parser.parse_args()

    if args.project_name:
        dataset_name = args.project_name
    if args.sub_dataset_name:
        sub_dataset_name = args.sub_dataset_name

    main()
