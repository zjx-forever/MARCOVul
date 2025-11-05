import concurrent.futures
import json
import random
import re
import threading

import Levenshtein
import re

statisticInfo = {}
all_data_path = 'allData.jsonl'
usable_data_id_file = 'DiverseVul2-usable_data.txt'

min_0_1_count = 1

multiple_restriction_0_1 = 3

whether_generateData_by_idx = False
whether_generateData = False


def generateData_jsonl_by_idx():
    train_file = open('train_base.jsonl', 'r')
    valid_file = open('valid_base.jsonl', 'r')
    test_file = open('test_base.jsonl', 'r')
    all_data = []
    for line in train_file.readlines():
        all_data.append(json.loads(line))
    for line in valid_file.readlines():
        all_data.append(json.loads(line))
    for line in test_file.readlines():
        all_data.append(json.loads(line))

    all_data = sorted(all_data, key=lambda x: x['idx'])

    global usable_data_id_file
    usable_data_file = open(usable_data_id_file, 'r')
    usable_data_id = set()
    for line in usable_data_file.readlines():
        usable_data_id.add(line.strip())

    with open(all_data_path, 'w') as allProjectFile:
        for item in all_data:
            if str(item['idx']) in usable_data_id:
                allProjectFile.write(json.dumps(item) + '\n')


def generateData_jsonl(data_file_json):
    file = open(data_file_json, 'r')
    items = []
    for line in file.readlines():
        dic = json.loads(line)
        items.append(dic)

    with open(all_data_path, 'w') as allProjectFile:

        index = 0
        for js in items:
            newjs = {'idx': index, 'project': js['project'], 'func': js['func'], 'target': int(js['target']),
                     'cwe': js['cwe'], 'size': js['size'], 'message': js['message'], 'commit_id': js['commit_id'],
                     'hash': js['hash']}
            index += 1

            allProjectFile.write(json.dumps(newjs) + '\n')


def statistic_base_Info(items):
    projectSet = set()
    CWESet = set()

    statistic_base = {}
    statistic_project_vul = {}
    statistic_CWE = {}
    statistic_CWE_vul = {}
    statistic_CWE_project_vul = {}

    count_vul = 0
    count_novul = 0

    for js in items:
        projectSet.add(js['project'])

        if js['target'] == 1:
            count_vul += 1
        elif js['target'] == 0:
            count_novul += 1
        else:
            print(f'target error: {js["target"]}')

        key_project_vul = js['project'] + '-' + str(js['target'])

        if key_project_vul in statistic_project_vul:

            statistic_project_vul[key_project_vul] += 1
        else:

            statistic_project_vul[key_project_vul] = 1

        CWE_type_list = js['cwe']
        for cwe in CWE_type_list:
            CWESet.add(cwe)
            key_cwe = cwe
            key_cwe_vul = cwe + '-' + str(js['target'])
            key_cwe_project_vul = cwe + '-' + js['project'] + '-' + str(js['target'])
            if key_cwe in statistic_CWE:
                statistic_CWE[key_cwe] += 1
            else:
                statistic_CWE[key_cwe] = 1

            if key_cwe_vul in statistic_CWE_vul:
                statistic_CWE_vul[key_cwe_vul] += 1
            else:
                statistic_CWE_vul[key_cwe_vul] = 1

            if key_cwe_project_vul in statistic_CWE_project_vul:
                statistic_CWE_project_vul[key_cwe_project_vul] += 1
            else:
                statistic_CWE_project_vul[key_cwe_project_vul] = 1

    statistic_project_vul = dict(sorted(statistic_project_vul.items(), key=lambda item: item[0]))
    statistic_CWE = dict(sorted(statistic_CWE.items(), key=lambda item: int(item[0].split('-')[1])))
    statistic_CWE_vul = dict(
        sorted(statistic_CWE_vul.items(), key=lambda item: (item[0].split('-')[1], item[0].split('-')[2])))
    statistic_CWE_project_vul = dict(sorted(statistic_CWE_project_vul.items(), key=lambda item: (
        item[0].split('-')[1], item[0].split('-')[2], item[0].split('-')[3])))

    statistic_vulNumber = {
        'vul': count_vul,
        'novul': count_novul
    }

    statistic_base = {
        'statistic_CWE_type_num': len(CWESet),
        'statistic_CWE_type': statistic_CWE,
        'statistic_project_num': len(projectSet),
        'statistic_project_vul': statistic_project_vul,
        'statistic_vulNumber': statistic_vulNumber,
        'statistic_CWE_vul': statistic_CWE_vul,
        'statistic_CWE_vul_project': statistic_CWE_project_vul
    }

    return statistic_base


def judge_which_cwe_project_lack(item, train_cwe_project, valid_cwe_project, test_cwe_project):
    count_0_1_dict = {
        "train-count-0": [],
        "train-count-1": [],
        "valid-count-0": [],
        "valid-count-1": [],
        "test-count-0": [],
        "test-count-1": []
    }
    for cwe in item['cwe']:
        key = f"{cwe}-{item['project']}"
        for cwe_project, count_0, count_1 in [
            (train_cwe_project, "train-count-0", "train-count-1"),
            (valid_cwe_project, "valid-count-0", "valid-count-1"),
            (test_cwe_project, "test-count-0", "test-count-1")
        ]:
            if key in cwe_project:
                count_0_1_dict[count_0].append(cwe_project[key].count(0))
                count_0_1_dict[count_1].append(cwe_project[key].count(1))
            else:
                count_0_1_dict[count_0].append(0)
                count_0_1_dict[count_1].append(0)

    def check_same_value(my_list):
        return -1 if len(set(my_list)) == 1 else my_list.index(min(my_list))

    if item['target'] == 1:
        for count_key in ["train-count-1", "valid-count-1", "test-count-1"]:
            index = check_same_value(count_0_1_dict[count_key])
            if index != -1:
                return index
    elif item['target'] == 0:
        for count_key in ["train-count-0", "valid-count-0", "test-count-0"]:
            index = check_same_value(count_0_1_dict[count_key])
            if index != -1:
                return index

    return 0


def split_data_by_cwe_project(all_data):
    train_cwe_project = {}
    valid_cwe_project = {}
    test_cwe_project = {}
    dataset_train = []
    dataset_valid = []
    dataset_test = []

    random.shuffle(all_data)

    for item in all_data:

        target = item['target']

        train_type = None

        lack_index = judge_which_cwe_project_lack(item, train_cwe_project, valid_cwe_project, test_cwe_project)

        cwe = item['cwe'][lack_index]

        key = f"{cwe}-{item['project']}"
        train_cwe_project_count_0 = 0
        train_cwe_project_count_1 = 0
        valid_cwe_project_count_0 = 0
        valid_cwe_project_count_1 = 0
        test_cwe_project_count_0 = 0
        test_cwe_project_count_1 = 0

        if key in train_cwe_project:
            train_cwe_project_count_0 = train_cwe_project[key].count(0)
            train_cwe_project_count_1 = train_cwe_project[key].count(1)
        else:
            train_cwe_project[key] = []

        if key in valid_cwe_project:
            valid_cwe_project_count_0 = valid_cwe_project[key].count(0)
            valid_cwe_project_count_1 = valid_cwe_project[key].count(1)
        else:
            valid_cwe_project[key] = []

        if key in test_cwe_project:
            test_cwe_project_count_0 = test_cwe_project[key].count(0)
            test_cwe_project_count_1 = test_cwe_project[key].count(1)
        else:
            test_cwe_project[key] = []

        if target == 1:

            if train_cwe_project_count_1 < 8:
                train_type = 1
            elif valid_cwe_project_count_1 == 0:
                train_type = 2
            elif test_cwe_project_count_1 == 0:
                train_type = 3

            else:

                if train_cwe_project_count_1 / valid_cwe_project_count_1 < 8 and train_cwe_project_count_1 / test_cwe_project_count_1 < 8:
                    train_type = 1
                elif train_cwe_project_count_1 / valid_cwe_project_count_1 >= 8:
                    train_type = 2
                elif train_cwe_project_count_1 / test_cwe_project_count_1 >= 8:
                    train_type = 3
                else:
                    print("Something wrong! target=1")
                    print('train:', train_cwe_project_count_1)
                    print('valid:', valid_cwe_project_count_1)
                    print('test:', test_cwe_project_count_1)
                    print('-----------------------------------')

        elif target == 0:
            if train_cwe_project_count_0 < 8:
                train_type = 1
            elif valid_cwe_project_count_0 == 0:
                train_type = 2
            elif test_cwe_project_count_0 == 0:
                train_type = 3
            else:

                if train_cwe_project_count_0 / valid_cwe_project_count_0 < 8 and train_cwe_project_count_0 / test_cwe_project_count_0 < 8:
                    train_type = 1
                elif train_cwe_project_count_0 / valid_cwe_project_count_0 >= 8:
                    train_type = 2
                elif train_cwe_project_count_0 / test_cwe_project_count_0 >= 8:
                    train_type = 3
                else:
                    print("Something wrong! target=0")
                    print('train:', train_cwe_project_count_0)
                    print('valid:', valid_cwe_project_count_0)
                    print('test:', test_cwe_project_count_0)
                    print('-----------------------------------')

        if train_type == 1:
            dataset_train.append(item)
        elif train_type == 2:
            dataset_valid.append(item)
        elif train_type == 3:
            dataset_test.append(item)

        for cwe in item['cwe']:
            key = f"{cwe}-{item['project']}"
            if train_type == 1:
                if key not in train_cwe_project:
                    train_cwe_project[key] = []
                train_cwe_project[key].append(target)
            elif train_type == 2:
                if key not in valid_cwe_project:
                    valid_cwe_project[key] = []
                valid_cwe_project[key].append(target)
            elif train_type == 3:
                if key not in test_cwe_project:
                    test_cwe_project[key] = []
                test_cwe_project[key].append(target)

    return dataset_train, dataset_valid, dataset_test


def save_dataset(dataset, file_path):
    with open(file_path, 'w') as allProjectFile:
        for item in dataset:
            allProjectFile.write(json.dumps(item) + '\n')


def data_preprocess(all_data):
    global multiple_restriction_0_1

    new_all_data = [item for item in all_data if item['cwe']]

    cwe_project_vul = {}
    for item in new_all_data:
        for cwe in item['cwe']:
            key = cwe + '-' + item['project']
            if key in cwe_project_vul:
                cwe_project_vul[key].append(item['target'])
            else:
                cwe_project_vul[key] = [item['target']]

    cwe_project_vul = {key: value for key, value in cwe_project_vul.items() if
                       value.count(1) >= 3 and value.count(0) >= 3}

    delete_idx = []

    new_all_data = sorted(new_all_data, key=lambda x: len(x['cwe']), reverse=True)
    new_all_data_add = []
    for item in new_all_data:
        new_item_cwe = []
        for cwe in item['cwe']:
            key = cwe + '-' + item['project']
            if key in cwe_project_vul:
                vul_0_1 = cwe_project_vul[key]

                if vul_0_1.count(0) / vul_0_1.count(1) > multiple_restriction_0_1:
                    if item['target'] == 1:
                        new_item_cwe.append(cwe)
                    else:
                        cwe_project_vul[key].remove(0)
                else:
                    new_item_cwe.append(cwe)

        if len(new_item_cwe) == 0:
            delete_idx.append(item['idx'])
        else:

            item['cwe'] = new_item_cwe
            new_all_data_add.append(item)

    return new_all_data_add


def cal_similarity(func1, func2):
    return 1 - Levenshtein.distance(func1, func2) / max(len(func1), len(func2))


def max_similarity(func, func_list):
    max_sim = 0
    for func2 in func_list:
        sim = cal_similarity(func, func2)
        if sim > max_sim:
            max_sim = sim
    return max_sim


def deal_key_values(key, values, multiple_restriction_0_1):
    global min_0_1_count

    count_0 = len([x['target'] for x in values if x['target'] == 0])
    count_1 = len([x['target'] for x in values if x['target'] == 1])

    if count_0 < min_0_1_count or count_1 < min_0_1_count:

        for item in values:
            cwe_type = key.split('-%-')[0]
            with my_lock:
                delete_idx_cwe.setdefault(item['idx'], []).append(cwe_type)
        return

    if count_0 / count_1 > multiple_restriction_0_1:
        need_delete_num = count_0 - count_1 * multiple_restriction_0_1
        target_1_func_list = [x['func'] for x in values if x['target'] == 1]

        similarity_list = []
        for v_i in range(len(values)):
            if values[v_i]['target'] == 0:
                similarity = max_similarity(values[v_i]['func'], target_1_func_list)
                similarity_list.append((v_i, similarity))
        similarity_list = sorted(similarity_list, key=lambda x: x[1])
        delete_index = [sim[0] for sim in similarity_list[:need_delete_num]]
        for v_i in delete_index:
            item = values[v_i]

            cwe_type = key.split('-%-')[0]
            with my_lock:
                delete_idx_cwe.setdefault(item['idx'], []).append(cwe_type)


my_lock = threading.Lock()
delete_idx_cwe = {}


def data_preprocess_similarity(all_data):
    global multiple_restriction_0_1

    num_workers = 32

    delete_idx_cwe.clear()

    cwe_project_vul = {}
    for item in all_data:
        for cwe in item['cwe']:
            key = cwe + '-%-' + item['project']
            cwe_project_vul.setdefault(key, []).append(item)

    print(f'Start deal_key_values!', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        error_list = []
        for key, values in cwe_project_vul.items():
            future = executor.submit(deal_key_values, key, values, multiple_restriction_0_1)
            error_list.append(future)
        executor.shutdown()

        flag = False
        for error in error_list:
            result = error.result()
            if result is not None:
                flag = True
                print('-----Error in deal_key_values-----', flush=True)
                print(result, flush=True)
                print('-----Error in deal_key_values-----', flush=True)
        if flag:
            exit(1)
    print(f'End deal_key_values!', flush=True)

    new_all_data_add = []
    for item in all_data:
        if item['idx'] not in delete_idx_cwe:
            new_all_data_add.append(item)
        else:
            item['cwe'] = [cwe for cwe in item['cwe'] if cwe not in delete_idx_cwe[item['idx']]]
            if item['cwe']:
                new_all_data_add.append(item)

    return new_all_data_add


def data_preprocess_delete_equal_func(all_data):
    def strip_silence_func(func_text: str) -> str:
        if not isinstance(func_text, str):
            return ""
        return re.sub(r"\s+", "", func_text)

    func_dict = {}
    delete_idx = []
    for item in all_data:
        norm_func = strip_silence_func(item['func'])
        if norm_func in func_dict:
            prev_target, prev_idx, prev_cwe = func_dict[norm_func]
            curr_target = item['target']
            curr_cwe = item['cwe']

            if prev_target != curr_target:
                if curr_target == 1 and prev_target == 0:

                    func_dict[norm_func] = (curr_target, item['idx'], curr_cwe)
                    delete_idx.append(prev_idx)
                elif curr_target == 0 and prev_target == 1:

                    delete_idx.append(item['idx'])
                else:

                    delete_idx.append(item['idx'])
            else:

                prev_cnt = len(prev_cwe) if isinstance(prev_cwe, list) else 0
                curr_cnt = len(curr_cwe) if isinstance(curr_cwe, list) else 0

                if prev_cnt > 0 and curr_cnt > 0:
                    if curr_cnt < prev_cnt:

                        func_dict[norm_func] = (curr_target, item['idx'], curr_cwe)
                        delete_idx.append(prev_idx)
                    else:

                        delete_idx.append(item['idx'])
                else:

                    if prev_cnt == 0 and curr_cnt > 0:
                        func_dict[norm_func] = (curr_target, item['idx'], curr_cwe)
                        delete_idx.append(prev_idx)
                    else:
                        delete_idx.append(item['idx'])
        else:
            func_dict[norm_func] = (item['target'], item['idx'], item['cwe'])

    if len(delete_idx) == 0:
        print('No equal func!', flush=True)
        return all_data

    print(f'Delete equal func! Delete idx: {delete_idx}', flush=True)
    new_all_data = [item for item in all_data if item['idx'] not in delete_idx]

    return new_all_data


def delete_empty_func(all_data):
    reg = r'\w+\s*\(.*?\)[ \t\n\s\w]*\{[ \t\n]*((return[^;]*;?)|(return\s*;))?[ \t\n]*\}'

    reg_end = r'}[\s\t\n]*$'
    new_all_data = []
    for item in all_data:
        current_func = item['func']
        match_func_empty = re.search(reg, current_func)
        match_func_end = re.search(reg_end, current_func)
        if match_func_empty is None and match_func_end is not None:
            new_all_data.append(item)
    return new_all_data


if __name__ == '__main__':
    print('Begin!', flush=True)

    data_file_json = 'diversevul_20230702.json'
    if whether_generateData:
        print('Start generateData_jsonl!', flush=True)
        generateData_jsonl(data_file_json)
        print('End generateData_jsonl!', flush=True)
    if whether_generateData_by_idx:
        print('Start generateData_jsonl_by_idx!', flush=True)
        generateData_jsonl_by_idx()
        print('End generateData_jsonl_by_idx!', flush=True)

    file = open(all_data_path, 'r')
    all_data = []
    for line in file.readlines():
        data_dict = json.loads(line)
        all_data.append(data_dict)

    statistic_base_before_preprocess = statistic_base_Info(all_data)

    print('Start data_preprocess_delete_equal_func!', flush=True)
    all_data = data_preprocess_delete_equal_func(all_data)
    print('End data_preprocess_delete_equal_func!', flush=True)

    print('Start delete empty cwe!', flush=True)
    all_data = [item for item in all_data if item['cwe']]
    print('End delete empty cwe!', flush=True)

    print('Start delete_empty_func!', flush=True)
    all_data = delete_empty_func(all_data)
    print('End delete_empty_func!', flush=True)

    print('Start data_preprocess_similarity!', flush=True)
    all_data = data_preprocess_similarity(all_data)
    print('End data_preprocess_similarity!', flush=True)

    statistic_base_after_preprocess = statistic_base_Info(all_data)

    print('Start split_data_by_cwe_project!', flush=True)
    train_dataset, valid_dataset, test_dataset = split_data_by_cwe_project(all_data)

    save_dataset(train_dataset, './train.jsonl')
    save_dataset(valid_dataset, './valid.jsonl')
    save_dataset(test_dataset, './test.jsonl')
    print('End split_data_by_cwe_project!', flush=True)

    statistic_base_train = statistic_base_Info(train_dataset)
    statistic_base_valid = statistic_base_Info(valid_dataset)
    statistic_base_test = statistic_base_Info(test_dataset)

    statisticInfo.update({
        'statistic_base_before_preprocess': statistic_base_before_preprocess,
        'statistic_base_after_preprocess': statistic_base_after_preprocess,
        'statistic_base_train': statistic_base_train,
        'statistic_base_valid': statistic_base_valid,
        'statistic_base_test': statistic_base_test
    })

    with open('./statisticInfo.json', 'w') as statisticInfoFile:
        json.dump(statisticInfo, statisticInfoFile, indent=4)

    print('End!', flush=True)
