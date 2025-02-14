#!/bin/bash

# 定义项目文件夹路径
PROJECT_DIR="/workspace/codebug/codebugmodel/"
#PROJECT_DIR="D:/409-AIOps/Static Software Defect Prediction/codebugmodel"

DATE="2024-11-18"

DATASET_NAME="DiverseVul-Derived2"
SUB_DATASET_NAME="all"

SIMPLIFY_DATASET_NAME="$DATASET_NAME-simplify"
SIMPLIFY_ENHANCE_DATASET_NAME="$SIMPLIFY_DATASET_NAME-enhance"

## 衍生数据集构造
## Levenshtein Distance
#echo "Start to construct Derived data set!"
#cd "$PROJECT_DIR/dataset/$DATASET_NAME" || exit 1
#python "derivationAndPartitionDataset.py" > "derivationAndPartitionDataset-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "derivationAndPartitionDataset.py executed successfully!"
#else
#    echo "derivationAndPartitionDataset.py executed failed!"
#    exit 1  # 或者其他退出码，根据需求设定
#fi
#echo -e "End to construct Derived data set!\n"
#
#
#source activate joern
#
#
## 生成AST、CFG、PDG
#echo "Start to generate AST, CFG, PDG!"
#cd "$PROJECT_DIR/joern/" || exit 1
#python "joern_parse.py" -ds $DATASET_NAME  > "joern_parse-$DATE-$DATASET_NAME-$SUB_DATASET_NAME-ast-cfg-pdg.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "joern_parse.py executed successfully!"
#else
#    echo "joern_parse.py executed failed!"
#    exit 1
#fi
#echo -e "End to generate AST, CFG, PDG!\n"
#
#
#conda deactivate
#
#
source activate bugdetect
#
#
## 验证数据可用性，删除不可用数据
#echo "Start to valid data usable!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "validDataUsable.py" -p $DATASET_NAME -sub $SUB_DATASET_NAME  > "validDataUsable-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "validDataUsable.py executed successfully!"
#else
#    echo "validDataUsable.py executed failed!"
#    exit 1
#fi
#echo -e "End to valid data usable!\n"
#
#
#
## 简化AST、CFG、PDG
## 会自动生成相应的简化后数据集，生成pyg data文件
#echo "Start to simplify AST, CFG, PDG!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "simplifyMiddleStructureCode.py" -p $DATASET_NAME -sub $SUB_DATASET_NAME -n 32 > "simplifyMiddleStructureCode-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "simplifyMiddleStructureCode.py executed successfully!"
#else
#    echo "simplifyMiddleStructureCode.py executed failed!"
#    exit 1
#fi
#echo -e "End to simplify AST, CFG, PDG!\n"
#
#
## cp文件，保留过程数据
#echo "Start to cp dataEnhance data set!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "cpPickle.py" -s $SIMPLIFY_DATASET_NAME -t $SIMPLIFY_ENHANCE_DATASET_NAME > "cpPickle-$DATE-$SIMPLIFY_DATASET_NAME-$SIMPLIFY_ENHANCE_DATASET_NAME.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "cpPickle.py executed successfully!"
#else
#    echo "cpPickle.py executed failed!"
#    exit 1
#fi
#echo -e "End to cp dataEnhance data set!\n"
#
#
## 删除文件BySize，防止有的函数过大，模型爆显存
#echo "Start to delete file by size!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "deleteFileBySize.py" -r $SIMPLIFY_ENHANCE_DATASET_NAME -sub $SUB_DATASET_NAME -t pdg ast cfg -size 102400 > "deleteFileBySize-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME-100k.out" 2>&1 &
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "deleteFileBySize.py executed successfully!"
#else
#    echo "deleteFileBySize.py executed failed!"
#    exit 1
#fi
#echo -e "End to delete file by size!\n"
#
#
## 数据增强
#echo "Start to enhance data!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "dataEnhance.py" -p $SIMPLIFY_ENHANCE_DATASET_NAME -sub $SUB_DATASET_NAME > "dataEnhance-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "dataEnhance.py executed successfully!"
#else
#    echo "dataEnhance.py executed failed!"
#    exit 1
#fi
#echo -e "End to enhance data!\n"
#
#
## 统计输入模型的数据
#echo "Start to do statistics!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "statisticsCweProjectVul.py" -p $SIMPLIFY_ENHANCE_DATASET_NAME -sub $SUB_DATASET_NAME > "statistics-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "statisticsCweProjectVul.py executed successfully!"
#else
#    echo "statisticsCweProjectVul.py executed failed!"
#    exit 1
#fi
#echo -e "End to do statistics!\n"
#
#调整SIMPLIFY_ENHANCE_DATASET_NAME以测试不同数据TODO
#SIMPLIFY_ENHANCE_DATASET_NAME=$DATASET_NAME
#SIMPLIFY_ENHANCE_DATASET_NAME='DiverseVul-Derived2-enhance'


# 运行模型。TODO: 修改模型运行参数
#echo "Start to run model!"
##MODEL_NAME="GCN_GCN_RGCN_LLM"
##MODEL_NAME="GCN_GCN_RGCN_W2V"
##MODEL_NAME="GAT_GAT_RGAT_W2V"
##MODEL_NAME="GCN_GCN_RGCN_W2V_Single_Test"
##MODEL_NAME="GIN_GCN_RGCN_W2V"
##MODEL_NAME="GAT_GAT_RGAT_LLM"
#MODEL_NAME="MulModel_Single_Test_LLM"
##MODEL_NAME="Single_Text_LLM"
##MODEL_NAME="word2vec"
#
#OTHER_INFO="Derivate2-100KB--s-e--single-ast--Text--$MODEL_NAME--GIN--LLM---13"
##OTHER_INFO="Derivate2-100KB--s-e--single-Text--$MODEL_NAME--LLM---2"
#echo "Current model is $MODEL_NAME"
#echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
#cd "$PROJECT_DIR/MultiFCode/" || exit 1
##python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME -w2vp $SIMPLIFY_ENHANCE_DATASET_NAME -w2vsub $SUB_DATASET_NAME -w2vsg 1 > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
#python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME -w2vp $SIMPLIFY_ENHANCE_DATASET_NAME -w2vsub $SUB_DATASET_NAME -w2vsg 1 > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "train.py executed successfully!"
#else
#    echo "train.py executed failed!"
#    exit 1
#fi
#echo -e "End to run model!\n"




# base
# 运行模型。
echo "Start to run model!"
#MODEL_NAME="MulModel_Single_Test_LLM"
#MODEL_NAME="MulModel_Four_modules_LLM"
MODEL_NAME="Single_Text_LLM"
#MODEL_NAME="GCN_GCN_RGCN_LLM"

#ADD_INFO="-1-NOTEXT-3layer-1024"
#OTHER_INFO="Derivate2-100KB--s-e--ast-cfg-pdg--noText--$MODEL_NAME--GCN--LLM---1$ADD_INFO"
#OTHER_INFO="-s-e--ast-cfg-pdg--useText--$MODEL_NAME--TreeLSTM-GAT3h-GIN--LLM---81-pre_train_structure-MLP-att-self(2h)-n-concatTrue-1024-e2-0.8"
OTHER_INFO="-s-e--useText--$MODEL_NAME--LLM---100"

#OTHER_INFO="-s-e--ast-cfg-pdg--useText--$MODEL_NAME--TreeLSTM-GAT3h-GIN--LLM---76---mlp-e3-0.9-1-2-42972"
#OTHER_INFO="-s-e--ast-cfg-pdg--useText--$MODEL_NAME--TreeLSTM-GAT3h-GIN--LLM---69-pre_train_structure-MLP-old-best-e2-0.8-1-1"
#OTHER_INFO="-s-e--ast-cfg-pdg--useText--$MODEL_NAME--TreeLSTM-GAT3h-GIN--LLM---30-pre_train_structure-MLP-grad-self-att-old-view-1024-e2-0.9"
#OTHER_INFO="-s-e--useText--$MODEL_NAME--LLM---1"
#OTHER_INFO="Derivate2-100KB--s-e--cfg--noText--$MODEL_NAME--GAT--LLM---4-3head-concat"

#OTHER_INFO="Derivate2-100KB--s-e--pdg--noText--$MODEL_NAME--GIN--LLM---1---test"

echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME -w2vp $SIMPLIFY_ENHANCE_DATASET_NAME -w2vsub $SUB_DATASET_NAME -w2vsg 1 > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
# 检查退出状态码
if [ $? -eq 0 ]; then
    echo "train.py executed successfully!"
else
    echo "train.py executed failed!"
    exit 1
fi
echo -e "End to run model!\n"

#SIMPLIFY_ENHANCE_DATASET_NAME=$DATASET_NAME
#
#OTHER_INFO="Derivate2-100KB--ori--ast-cfg-pdg--noText--$MODEL_NAME--GCN--LLM---2$ADD_INFO"
#echo "Current model is $MODEL_NAME"
#echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
#cd "$PROJECT_DIR/MultiFCode/" || exit 1
#python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME -w2vp $SIMPLIFY_ENHANCE_DATASET_NAME -w2vsub $SUB_DATASET_NAME -w2vsg 1 > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "train.py executed successfully!"
#else
#    echo "train.py executed failed!"
#    exit 1
#fi
#echo -e "End to run model!\n"
#
#SIMPLIFY_ENHANCE_DATASET_NAME='DiverseVul-Derived2-enhance'
#
#OTHER_INFO="Derivate2-100KB--e--ast-cfg-pdg--noText--$MODEL_NAME--GCN--LLM---3$ADD_INFO"
#echo "Current model is $MODEL_NAME"
#echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
#cd "$PROJECT_DIR/MultiFCode/" || exit 1
#python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME -w2vp $SIMPLIFY_ENHANCE_DATASET_NAME -w2vsub $SUB_DATASET_NAME -w2vsg 1 > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "train.py executed successfully!"
#else
#    echo "train.py executed failed!"
#    exit 1
#fi
#echo -e "End to run model!\n"
#
#SIMPLIFY_ENHANCE_DATASET_NAME=$SIMPLIFY_DATASET_NAME
#
#OTHER_INFO="Derivate2-100KB--s--ast-cfg-pdg--noText--$MODEL_NAME--GCN--LLM---4$ADD_INFO"
#echo "Current model is $MODEL_NAME"
#echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
#cd "$PROJECT_DIR/MultiFCode/" || exit 1
#python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME -w2vp $SIMPLIFY_ENHANCE_DATASET_NAME -w2vsub $SUB_DATASET_NAME -w2vsg 1 > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
## 检查退出状态码
#if [ $? -eq 0 ]; then
#    echo "train.py executed successfully!"
#else
#    echo "train.py executed failed!"
#    exit 1
#fi
#echo -e "End to run model!\n"

conda deactivate