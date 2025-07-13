#!/bin/bash

PROJECT_DIR="/workspace/codebug/codebugmodel/"

DATE="2025-06-29"

DATASET_NAME="DiverseVul-Derived2"
SUB_DATASET_NAME="all"

SIMPLIFY_DATASET_NAME="$DATASET_NAME-simplify"
ENHANCE_DATASET_NAME="$DATASET_NAME-enhance"
SIMPLIFY_ENHANCE_DATASET_NAME="$SIMPLIFY_DATASET_NAME-enhance"


echo "Start to construct Derived data set!"
cd "$PROJECT_DIR/dataset/$DATASET_NAME" || exit 1
python "derivationAndPartitionDataset.py" > "derivationAndPartitionDataset-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

if [ $? -eq 0 ]; then
    echo "derivationAndPartitionDataset.py executed successfully!"
else
    echo "derivationAndPartitionDataset.py executed failed!"
    exit 1
fi
echo -e "End to construct Derived data set!\n"


source activate joern

echo "Start to generate AST, CFG, PDG!"
cd "$PROJECT_DIR/joern/" || exit 1
python "joern_parse.py" -ds $DATASET_NAME  > "joern_parse-$DATE-$DATASET_NAME-$SUB_DATASET_NAME-ast-cfg-pdg.out" 2>&1

if [ $? -eq 0 ]; then
    echo "joern_parse.py executed successfully!"
else
    echo "joern_parse.py executed failed!"
    exit 1
fi
echo -e "End to generate AST, CFG, PDG!\n"

conda deactivate


source activate bugdetect

echo "Start to valid data usable!"
cd "$PROJECT_DIR/joern/data" || exit 1
python "validDataUsable.py" -p $DATASET_NAME -sub $SUB_DATASET_NAME  > "validDataUsable-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

if [ $? -eq 0 ]; then
    echo "validDataUsable.py executed successfully!"
else
    echo "validDataUsable.py executed failed!"
    exit 1
fi
echo -e "End to valid data usable!\n"


echo "Start to simplify AST, CFG, PDG!"
cd "$PROJECT_DIR/joern/data" || exit 1
python "simplifyMiddleStructureCode.py" -p $DATASET_NAME -sub $SUB_DATASET_NAME -n 32 > "simplifyMiddleStructureCode-$DATE-$DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

if [ $? -eq 0 ]; then
    echo "simplifyMiddleStructureCode.py executed successfully!"
else
    echo "simplifyMiddleStructureCode.py executed failed!"
    exit 1
fi
echo -e "End to simplify AST, CFG, PDG!\n"


echo "Start to cp dataEnhance data set!"
cd "$PROJECT_DIR/joern/data" || exit 1
python "cpPickle.py" -s $SIMPLIFY_DATASET_NAME -t $SIMPLIFY_ENHANCE_DATASET_NAME > "cpPickle-$DATE-$SIMPLIFY_DATASET_NAME-$SIMPLIFY_ENHANCE_DATASET_NAME.out" 2>&1

if [ $? -eq 0 ]; then
    echo "cpPickle.py executed successfully!"
else
    echo "cpPickle.py executed failed!"
    exit 1
fi
echo -e "End to cp dataEnhance data set!\n"


#echo "Start to delete file by size!"
#cd "$PROJECT_DIR/joern/data" || exit 1
#python "deleteFileBySize.py" -r $SIMPLIFY_ENHANCE_DATASET_NAME -sub $SUB_DATASET_NAME -t pdg ast cfg -size 102400 > "deleteFileBySize-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME-100k.out" 2>&1 &
#
#if [ $? -eq 0 ]; then
#    echo "deleteFileBySize.py executed successfully!"
#else
#    echo "deleteFileBySize.py executed failed!"
#    exit 1
#fi
#echo -e "End to delete file by size!\n"


echo "Start to enhance data!"
cd "$PROJECT_DIR/joern/data" || exit 1
python "dataEnhance.py" -p $SIMPLIFY_ENHANCE_DATASET_NAME -sub $SUB_DATASET_NAME > "dataEnhance-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

if [ $? -eq 0 ]; then
    echo "dataEnhance.py executed successfully!"
else
    echo "dataEnhance.py executed failed!"
    exit 1
fi
echo -e "End to enhance data!\n"


echo "Start to do statistics!"
cd "$PROJECT_DIR/joern/data" || exit 1
python "statisticsCweProjectVul.py" -p $SIMPLIFY_ENHANCE_DATASET_NAME -sub $SUB_DATASET_NAME > "statistics-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME.out" 2>&1

if [ $? -eq 0 ]; then
    echo "statisticsCweProjectVul.py executed successfully!"
else
    echo "statisticsCweProjectVul.py executed failed!"
    exit 1
fi
echo -e "End to do statistics!\n"


echo "-------------------------------------------------------------------------------"
echo "Start to run model!"
echo "Start to run four model-s-e! ---MARCOVul!"
echo "-------------------------------------------------------------------------------"
MODEL_NAME="MulModel_Four_modules_LLM"

echo "Current model is $MODEL_NAME"
OTHER_INFO="-s-e--ast-cfg-pdg-useText--$MODEL_NAME--LLM---1-MARCOVul"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "ast" "cfg" "pdg" -ut "true" -pe "true" -ptsu "true" -ptse "false" -wr "false"\
> "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "-------------------------------------------------------------------------------"
echo "Run Orig-Simp-Aug-SimpAug Test! ---RQ2---Table III"
echo "-------------------------------------------------------------------------------"
echo "Start to run s-e Test!"

MODEL_NAME="GCN_GCN_RGCN_LLM"

echo "Current is s-e!"
OTHER_INFO="-GCN-s-e--ast-cfg-pdg-noText--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "ast" "cfg" "pdg" -ut "false" -pe "true" \
> "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "Current is ori!"
OTHER_INFO="-GCN-ori--ast-cfg-pdg-noText--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "ast" "cfg" "pdg" -ut "false" -pe "true" -wr "false"\
> "train-$DATE-$DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "Current is s!"
OTHER_INFO="-GCN-s--ast-cfg-pdg-noText--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "ast" "cfg" "pdg" -ut "false" -pe "true" -wr "false"\
> "train-$DATE-$SIMPLIFY_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "Current is e!"
OTHER_INFO="-GCN-e--ast-cfg-pdg-noText--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "ast" "cfg" "pdg" -ut "false" -pe "true" -wr "false"\
> "train-$DATE-$ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"


echo "-------------------------------------------------------------------------------"
echo "Start to run four model! ---Ablation---Table V"
echo "-------------------------------------------------------------------------------"
MODEL_NAME="MulModel_Four_modules_LLM"

echo "Current is ast-cfg-pdg-noText!"
OTHER_INFO="-s-e--ast-cfg-pdg-noText--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "ast" "cfg" "pdg" -ut "false" -pe "true" -ptsu "false" -ptse "false" \
> "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "Current is ast!"
OTHER_INFO="-s-e--ast--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "ast" -ut "false" -pe "true" -ptsu "false" -ptse "false" \
> "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "Current is cfg!"
OTHER_INFO="-s-e--cfg--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "cfg" -ut "false" -pe "true" -ptsu "false" -ptse "false" \
> "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "Current is pdg!"
OTHER_INFO="-s-e--pdg--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "pdg" -ut "false" -pe "true" -ptsu "false" -ptse "false" \
> "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "Current is Text!"
MODEL_NAME="Single_Text_LLM"
OTHER_INFO="-s-e--Text--$MODEL_NAME--LLM---1"
echo "Current model is $MODEL_NAME"
echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
cd "$PROJECT_DIR/MultiFCode/" || exit 1
python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
-gt "ast" -ut "true" -pe "true" -ptsu "false" -ptse "false" \
> "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1

if [ $? -eq 0 ]; then
   echo "train.py executed successfully!"
else
   echo "train.py executed failed!"
   exit 1
fi
echo -e "End to run model!\n"

echo "-------------------------------------------------------------------------------"
echo "Start to run single model Test! ---RQ4"
echo "-------------------------------------------------------------------------------"
MODEL_NAME="MulModel_Single_Test_LLM"

modelNames_AST=("GIN" "TreeLSTM" "GCN" "GAT" "FILM" "GMM" "TransformerConv" "TAG" "ResGatedGraphConv" "SAGE")
modelNames_CFG_PDG=("GIN" "GCN" "GAT" "FILM" "GMM" "TransformerConv" "TAG" "ResGatedGraphConv" "SAGE")

echo "Start to run AST Test!"

for CURRENT_MODEL_NAME in "${modelNames_AST[@]}"; do
 OTHER_INFO="-s-e--AST-$CURRENT_MODEL_NAME--$MODEL_NAME--LLM---1"
 echo "Current model is $MODEL_NAME"
 echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
 cd "$PROJECT_DIR/MultiFCode/" || exit 1
 python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
 -gt "ast" -ut "false" -pe "true" -ptsu "false" -gtmAST "$CURRENT_MODEL_NAME" \
 > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
 
 if [ $? -eq 0 ]; then
     echo "train.py executed successfully!"
 else
     echo "train.py executed failed!"
     exit 1
 fi
 echo -e "End to run model!\n"
done

echo "Start to run CFG Test!"

for CURRENT_MODEL_NAME in "${modelNames_CFG_PDG[@]}"; do
  OTHER_INFO="-s-e--CFG-$CURRENT_MODEL_NAME--$MODEL_NAME--LLM---1"
  echo "Current model is $MODEL_NAME"
  echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
  cd "$PROJECT_DIR/MultiFCode/" || exit 1
  python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
  -gt "cfg" -ut "false" -pe "true" -ptsu "false" -gtmCFG "$CURRENT_MODEL_NAME" \
  > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
  
  if [ $? -eq 0 ]; then
      echo "train.py executed successfully!"
  else
      echo "train.py executed failed!"
      exit 1
  fi
  echo -e "End to run model!\n"
done

echo "Start to run PDG Test!"

for CURRENT_MODEL_NAME in "${modelNames_CFG_PDG[@]}"; do
  OTHER_INFO="-s-e--PDG-$CURRENT_MODEL_NAME--$MODEL_NAME--LLM---1"
  echo "Current model is $MODEL_NAME"
  echo "Current dataset is $SIMPLIFY_ENHANCE_DATASET_NAME"
  cd "$PROJECT_DIR/MultiFCode/" || exit 1
  python "train.py" -model $MODEL_NAME -dsn $SIMPLIFY_ENHANCE_DATASET_NAME -dssub $SUB_DATASET_NAME \
  -gt "pdg" -ut "false" -pe "true" -ptsu "false" -gtmPDG "$CURRENT_MODEL_NAME" \
  > "train-$DATE-$SIMPLIFY_ENHANCE_DATASET_NAME-$SUB_DATASET_NAME--$OTHER_INFO.out" 2>&1
  
  if [ $? -eq 0 ]; then
      echo "train.py executed successfully!"
  else
      echo "train.py executed failed!"
      exit 1
  fi
  echo -e "End to run model!\n"
done

conda deactivate