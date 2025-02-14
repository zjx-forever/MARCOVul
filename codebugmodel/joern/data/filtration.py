import json



def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    
    data = read_json('statisticInfo_all_cwe_project_vul_num.json')
    eq0_list = []
    for key in data:
        for sub_key in data[key]:
            if data[key][sub_key] == 0:
                new_string = key + '---' + sub_key
                eq0_list.append(new_string)

    with open('eq0_list.txt', 'w') as f:
        for item in eq0_list:
            f.write(item + '\n')

    train_0_key = []
    for key in data:
        if key.endswith('train'):
            cwe_projects = data[key]
            for sub_key in cwe_projects:
                if cwe_projects[sub_key] == 0:
                    train_0_key.append(sub_key)
    for key in data:
        if key.endswith('valid') or key.endswith('test'):
            cwe_projects = data[key]
            for sub_key in cwe_projects:
                if sub_key in train_0_key and cwe_projects[sub_key] != 0:
                    print(key, sub_key)
                    print(cwe_projects[sub_key])
                    print('---------------------------------')


