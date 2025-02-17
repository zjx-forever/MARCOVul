import argparse
import os
import shutil

sourceFileName = 'xxx'
targetFileName = 'xxx'

includeName = '_output_pickle_'


def copyFileWithStructure(sourceFile, targetFile, include):
    all_sub_projects_path = []
    for dirname in os.listdir(sourceFile):

        dirpath = os.path.join(sourceFile, dirname)

        if os.path.isdir(dirpath):
            all_sub_projects_path.append(dirpath)
    for sub_project_path in all_sub_projects_path:
        for dirname in os.listdir(sub_project_path):

            dirpath = os.path.join(sub_project_path, dirname)

            if os.path.isdir(dirpath) and include in dirname:
                newRoot = replaceFirstPath(sub_project_path, targetFile)
                shutil.copytree(os.path.join(sub_project_path, dirname), os.path.join(newRoot, dirname))
                print('copy ' + os.path.join(sub_project_path, dirname) + ' to ' + os.path.join(newRoot, dirname))


def replaceFirstPath(path, newFirstPath):
    pathList = path.split(os.path.sep)
    pathList[0] = newFirstPath
    return os.path.sep.join(pathList)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sourceFileName', help='sourceFileName')
    parser.add_argument('-t', '--targetFileName', help='targetFileName')
    parser.add_argument('-include', '--includeName', help='includeName')

    args = parser.parse_args()

    if args.sourceFileName:
        sourceFileName = args.sourceFileName
    if args.targetFileName:
        targetFileName = args.targetFileName
    if args.includeName:
        includeName = args.includeName

    copyFileWithStructure(sourceFileName, targetFileName, includeName)
