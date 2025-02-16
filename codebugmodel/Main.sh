#!/bin/bash

PROJECT_DIR="/workspace/codebug/codebugmodel/"

DATE="2024-11-18"

DATASET_NAME="DiverseVul-Derived"
SUB_DATASET_NAME="all"

SIMPLIFY_DATASET_NAME="$DATASET_NAME-simplify"
SIMPLIFY_ENHANCE_DATASET_NAME="$SIMPLIFY_DATASET_NAME-enhance"

## Levenshtein Distance
#echo "Start to construct Derived data set!"
#cd "$PROJECT_DIR/dataset/$DATASET_NAME" || exit 1
#python "derivationAndPartitionDataset.py" > "derivationAndPartitionDataset-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

#if [ $? -eq 0 ]; then
#    echo "derivationAndPartitionDataset.py executed successfully!"
#else
#    echo "derivationAndPartitionDataset.py executed failed!"
#    exit 1  
#fi
#echo -e "End to construct Derived data set!\n"
#
#
#source activate joern
#
#
## gen AST、CFG、PDG
#echo "Start to generate AST, CFG, PDG!"
#cd "$PROJECT_DIR/joern/" || exit 1
#python "joern_parse.py" -ds $DATASET_NAME  > "joern_parse-$DATE-$DATASET_NAME-$SUB_DATASET_NAME-ast-cfg-pdg.out" 2>&1

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
#echo "Start to valid data usable!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "validDataUsable.py" -p $DATASET_NAME -sub $SUB_DATASET_NAME  > "validDataUsable-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

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
#echo "Start to simplify AST, CFG, PDG!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "simplifyMiddleStructureCode.py" -p $DATASET_NAME -sub $SUB_DATASET_NAME -n 32 > "simplifyMiddleStructureCode-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

#if [ $? -eq 0 ]; then
#    echo "simplifyMiddleStructureCode.py executed successfully!"
#else
#    echo "simplifyMiddleStructureCode.py executed failed!"
#    exit 1
#fi
#echo -e "End to simplify AST, CFG, PDG!\n"
#
#
#echo "Start to cp dataEnhance data set!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "cpPickle.py" -s $SIMPLIFY_DATASET_NAME -t $SIMPLIFY_ENHANCE_DATASET_NAME > "cpPickle-$DATE-$SIMPLIFY_DATASET_NAME-$SIMPLIFY_ENHANCE_DATASET_NAME.out" 2>&1

#if [ $? -eq 0 ]; then
#    echo "cpPickle.py executed successfully!"
#else
#    echo "cpPickle.py executed failed!"
#    exit 1
#fi
#echo -e "End to cp dataEnhance data set!\n"
#
#echo "Start to enhance data!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "dataEnhance.py" -p $SIMPLIFY_ENHANCE_DATASET_NAME -sub $SUB_DATASET_NAME > "dataEnhance-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

#if [ $? -eq 0 ]; then
#    echo "dataEnhance.py executed successfully!"
#else
#    echo "dataEnhance.py executed failed!"
#    exit 1
#fi
#echo -e "End to enhance data!\n"
#
#
#echo "Start to do statistics!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "statisticsCweProjectVul.py" -p $SIMPLIFY_ENHANCE_DATASET_NAME -sub $SUB_DATASET_NAME > "statistics-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

#if [ $? -eq 0 ]; then
#    echo "statisticsCweProjectVul.py executed successfully!"
#else
#    echo "statisticsCweProjectVul.py executed failed!"
#    exit 1
#fi
#echo -e "End to do statistics!\n"
#

# base
echo "Start to run model!"
#MODEL_NAME="GCN_GCN_RGCN_LLM"
#MODEL_NAME="GAT_GAT_RGAT_LLM"
MODEL_NAME="Single_Text_LLM"
# MODEL_NAME="MulModel_Four_modules_LLM"

OTHER_INFO="-s-e--useText--$MODEL_NAME--LLM---100"

echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME -w2vp $SIMPLIFY_ENHANCE_DATASET_NAME -w2vsub $SUB_DATASET_NAME -w2vsg 1 > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
if [ $? -eq 0 ]; then
    echo "train.py executed successfully!"
else
    echo "train.py executed failed!"
    exit 1
fi
echo -e "End to run model!\n"

conda deactivate