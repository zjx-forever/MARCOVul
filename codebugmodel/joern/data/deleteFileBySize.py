import argparse
import os
import shutil


root_folder_path = 'xxx'


size_threshold = 100 * 1024  


def delete_large_files_in_codeTypeList_and_merge(subProjects, codeTypeList):
    
    delete_end = ['train', 'valid', 'test']
    for subProject in subProjects:
        for end in delete_end:
            delete_and_merge(subProject, end, codeTypeList)

def delete_and_merge(subProject, end_with, codeTypeList):
    needString = 'output_pickle'
    
    need_delete_id = set()

    
    type_id_map = {}
    for t in codeTypeList:
        type_id_map[t] = set()

    
    for root, dirs, files in os.walk(root_folder_path):
        if subProject in root:
            
            if os.path.basename(root).endswith(end_with) and needString in root:
                dir_type = root.split(os.path.sep)[-1].split('_')[0]
                for file in files:
                    file_path = os.path.join(root, file)
                    file_id = file.split('-')[1]
                    
                    type_id_map[dir_type].add(file_id)
                    
                    file_size = os.path.getsize(file_path)
                    
                    if file_size > size_threshold:
                        need_delete_id.add(file_id)

    
    common_id = type_id_map[codeTypeList[0]]
    for t in codeTypeList:
        common_id = common_id & type_id_map[t]

    
    for root, dirs, files in os.walk(root_folder_path):
        if subProject in root:
            
            if os.path.basename(root).endswith(end_with) and needString in root:
                for file in files:
                    file_path = os.path.join(root, file)
                    file_id = file.split('-')[1]
                    if file_id in need_delete_id or file_id not in common_id:
                        print(f"Deleting {file_path}")
                        os.remove(file_path)


if __name__ == '__main__':
    
    subProjectName = ['xxx1', 'xxx2', 'all']
    typeList = ['pdg', 'ast', 'cfg']

    
    parser = argparse.ArgumentParser()

    
    parser.add_argument('-r', '--root_folder_path', help='root_folder_path')
    parser.add_argument('-sub', '--subProjects', nargs='+', help='subProjectName')
    parser.add_argument('-size', '--size_threshold', type=int, help='size_threshold')
    parser.add_argument('-t', '--typeList', nargs='+', help='typeList')

    
    args = parser.parse_args()

    if args.root_folder_path:
        root_folder_path = args.root_folder_path
    if args.subProjects:
        subProjectName = args.subProjects
    if args.size_threshold:
        size_threshold = args.size_threshold
    if args.typeList:
        typeList = args.typeList

    print('Start')
    delete_large_files_in_codeTypeList_and_merge(subProjectName, typeList)
    print('Done')
