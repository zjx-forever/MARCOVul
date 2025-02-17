import argparse
import sys
import os

import multiprocessing
from multiprocessing import cpu_count, Manager, Pool, Queue
import subprocess
import time
from typing import cast

import pygraphviz as pgv
import pickle
import networkx as nx
import json

from omegaconf import DictConfig, OmegaConf

from symbolizer import clean_gadget

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.parse_args import configure_arg_parser

USE_CPU = cpu_count()

current_file_directory = os.path.dirname(os.path.abspath(__file__))

parent_directory = os.path.dirname(current_file_directory)


def run_command_with_retries(command, max_retries=10):
    attempt = 0
    while attempt < max_retries:
        result = subprocess.run(command, shell=True)

        if result.returncode == 0:
            return result
        else:
            print('-' * 10)
            print(
                f"The command-{command}-execution failed, with a return code of {result.returncode}, and the attempt count is now {attempt + 1}")
            print('-' * 10)
            attempt += 1
            time.sleep(5)
    print('-' * 10)
    print(f"The command-{command}-failed to execute multiple times, reaching the maximum number of attempts.")
    print('-' * 10)
    return None


def generate_dir(config: DictConfig, split_name, datasetName, projectName, generate_type):
    data_dir = config.data_folder

    temp_function_dir_path = os.path.join(current_file_directory, data_dir, datasetName, projectName,
                                          f'{generate_type}_code_{split_name}')

    output_dir_path = os.path.join(current_file_directory, data_dir, datasetName, projectName,
                                   f'{generate_type}_output_{split_name}')

    output_dot_max_path = os.path.join(current_file_directory, data_dir, datasetName, projectName,
                                       f'{generate_type}_output_dot_{split_name}')

    output_pickle_path = os.path.join(current_file_directory, data_dir, datasetName, projectName,
                                      f'{generate_type}_output_pickle_{split_name}')

    if os.path.exists(temp_function_dir_path):
        os.system(f'rm -rf {temp_function_dir_path}')
    if os.path.exists(output_dir_path):
        os.system(f'rm -rf {output_dir_path}')
    if os.path.exists(output_dot_max_path):
        os.system(f'rm -rf {output_dot_max_path}')
    if os.path.exists(output_pickle_path):
        os.system(f'rm -rf {output_pickle_path}')

    os.makedirs(temp_function_dir_path, exist_ok=True)
    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(output_dot_max_path, exist_ok=True)
    os.makedirs(output_pickle_path, exist_ok=True)

    return temp_function_dir_path, output_dir_path, output_dot_max_path, output_pickle_path


def generate_pdg(config: DictConfig, js, clean_func, temp_function_dir_path, output_dir_path,
                 output_dot_max_path, output_pickle_path, export_format):
    try:
        idx = js['idx']
        target = js['target']
        func = js['func']

        joern_path_parse = config.joern_path_parse
        joern_path_export = config.joern_path_export

        export_type = config.export_type

        generate_file_suffix = config.generate_file_suffix

        temp_function_file_name = f'temp_function-{idx}-{target}'
        temp_function_file_name_suffix = f'temp_function-{idx}-{target}{generate_file_suffix}'
        temp_function_file_path = os.path.join(temp_function_dir_path, temp_function_file_name_suffix)

        final_output_dot_file_name = f'{export_format}-{idx}-{target}.{export_type}'

        with open(temp_function_file_path, 'w') as file:
            file.write(clean_func)

        cpg_file_path = os.path.join(output_dir_path, f'cpg-{temp_function_file_name}.bin')
        parse_command = f'{joern_path_parse} {temp_function_file_path} --output {cpg_file_path}'

        output_dot = os.path.join(output_dir_path, f'out-{temp_function_file_name}')
        export_command = f'{joern_path_export} {cpg_file_path} --repr {export_format} --out {output_dot} --format {export_type}'

        command_parse = f'{parse_command}'
        command_parse_return = run_command_with_retries(command_parse)
        if command_parse_return is None:
            print('-' * 10)
            print(f'Error: {parse_command} has problem')
            print('-' * 10)
            return

        command_export = f'{export_command}'
        command_export_return = run_command_with_retries(command_export)
        if command_export_return is None:
            print('-' * 10)
            print(f'Error: {export_command} has problem')
            print('-' * 10)
            return

        output_dot_files = os.listdir(output_dot)

        output_dot_files_size = [os.path.getsize(os.path.join(output_dot, file)) for file in output_dot_files]

        output_dot_max_file_index = output_dot_files_size.index(max(output_dot_files_size))

        output_dot_max_file_name = output_dot_files[output_dot_max_file_index]

        output_dot_max_file_path = os.path.join(output_dot, output_dot_max_file_name)

        if os.path.getsize(output_dot_max_file_path) != 32:

            new_file_name_all_path = os.path.join(output_dot_max_path, final_output_dot_file_name)
            command_cp = f'cp {output_dot_max_file_path} {new_file_name_all_path}'
            command_cp_return = run_command_with_retries(command_cp)
            if command_cp_return is None:
                print('-' * 10)
                print(f'Error: {command_cp} has problem')
                print('-' * 10)
                return

            graph = pgv.AGraph(new_file_name_all_path)
            G = nx.DiGraph(graph)
            G.graph['label'] = target
            G.graph['func'] = func
            G.graph['idx'] = idx
            if 'cwe' in js:
                G.graph['cwe'] = js['cwe']
            if 'project' in js:
                G.graph['project'] = js['project']

            pickle_file_name = f'{export_format}-{idx}-{target}.pkl'
            pickle_file_path = os.path.join(output_pickle_path, pickle_file_name)
            pickle.dump(G, open(pickle_file_path, 'wb'))


        else:
            print(f'Error: {output_dot_max_file_path} has problem')
    except Exception as e:
        print(f'Exception Error: {e}')


def readJSONDataAndGeneratePDG(config: DictConfig, datasetName, export_format):
    global max_processes

    pool = Pool(max_processes)
    print("CPU core num:", USE_CPU)
    split_name = ['test', 'valid', 'train']

    for splitName in split_name:
        temp_function_dir_path, output_dir_path, output_dot_max_path, output_pickle_path = generate_dir(config,
                                                                                                        splitName,
                                                                                                        datasetName,
                                                                                                        'all',
                                                                                                        export_format)
        with open(os.path.join(parent_directory, 'dataset', datasetName, f'{splitName}.jsonl'), 'r') as f:
            for line in f:
                js = json.loads(line)
                func = js['func']

                clean_func = clean_gadget([func])[0]

                pool.apply_async(generate_pdg,
                                 (config, js, clean_func, temp_function_dir_path, output_dir_path,
                                  output_dot_max_path, output_pickle_path, export_format))

    pool.close()

    pool.join()


def readJSONDataAndGeneratePDG_projects(config: DictConfig, datasetName, projectsName, export_format):
    global max_processes

    pool = Pool(max_processes)
    print("CPU core num:", USE_CPU)
    split_name = ['test', 'train', 'valid']

    for sub in projectsName:
        for splitName in split_name:
            temp_function_dir_path, output_dir_path, output_dot_max_path, output_pickle_path = generate_dir(config,
                                                                                                            splitName,
                                                                                                            datasetName,
                                                                                                            sub,
                                                                                                            export_format)
            with open(os.path.join(parent_directory, 'dataset', datasetName, sub, f'{splitName}.jsonl'), 'r') as f:
                for line in f:
                    js = json.loads(line)
                    func = js['func']

                    clean_func = clean_gadget([func])[0]

                    pool.apply_async(generate_pdg,
                                     (config, js, clean_func, temp_function_dir_path, output_dir_path,
                                      output_dot_max_path, output_pickle_path, export_format))

    pool.close()
    pool.join()


dataset_name = None
projects_name = None
max_processes = 64


def main():
    print("-----------------BEGIN main-----------------")
    arg_parser = configure_arg_parser()
    args, unknown = arg_parser.parse_known_args()
    config = cast(DictConfig, OmegaConf.load(args.config))
    global dataset_name

    export_format_list = list(config.joern.export_format)
    for export_format in export_format_list:
        readJSONDataAndGeneratePDG(config.joern, dataset_name, export_format)

    print("---------------------END---------------------")


def main_projects():
    print("-----------------BEGIN main_projects-----------------")
    arg_parser = configure_arg_parser()
    args, unknown = arg_parser.parse_known_args()
    config = cast(DictConfig, OmegaConf.load(args.config))
    global dataset_name, projects_name

    export_format_list = list(config.joern.export_format)
    for export_format in export_format_list:
        readJSONDataAndGeneratePDG_projects(config.joern, dataset_name, projects_name, export_format)

    print("-------------------------END-------------------------")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset_name', help='dataset_name')
    parser.add_argument('-sub', '--sub_project_name', help='sub_project_name')

    args = parser.parse_args()

    if args.dataset_name:
        dataset_name = args.dataset_name
    if args.sub_project_name:
        projects_name = args.sub_project_name

    main()
    if projects_name is not None:
        main_projects()
