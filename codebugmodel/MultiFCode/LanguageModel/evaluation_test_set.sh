#!/bin/bash
PROJECT_DIR="/workspace/codebug/codebugmodel"

source activate python3.9
cd "$PROJECT_DIR/MultiFCode/LanguageModel" || exit 1

echo "-----------------------------------------------------------------------------------------------------------------"
echo "Train olddataset and test dataset"
echo "-----------------------------------------------------------------------------------------------------------------"

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./oldDataset/train.jsonl \
    --eval_data_file=./oldDataset/valid.jsonl \
    --test_data_file=./oldDataset/test.jsonl \
    --output_dir=./saved_models/Derived2/oldDataset/codeBERT/microsoft-codebert-base/novar-1weight \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base \
    --cache_dir=./cache/codeBERT \
    --loss_weight 1 \
    --do_test \
    --seed 422 > codeBERT-microsoft-codebert-base-novar-train_old_test_old-1weight_evaluation-Derived2.out  2>&1
if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-novar-train_old_test_old-1weight_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-novar-train_old_test_old-1weight_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./oldDataset/train.jsonl \
    --eval_data_file=./oldDataset/valid.jsonl \
    --test_data_file=./oldDataset/test.jsonl \
    --output_dir=./saved_models/Derived2/oldDataset/codeBERT/microsoft-codebert-base/novar-20weight \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base \
    --cache_dir=./cache/codeBERT \
    --loss_weight 20 \
    --do_test \
    --seed 422 > codeBERT-microsoft-codebert-base-novar-train_old_test_old-20weight_evaluation-Derived2.out  2>&1
if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-novar-train_old_test_old-20weight_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-novar-train_old_test_old-20weight_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./oldDataset/train.jsonl \
    --eval_data_file=./oldDataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/oldDataset/codeBERT/microsoft-codebert-base/novar-1weight \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base \
    --cache_dir=./cache/codeBERT \
    --loss_weight 1 \
    --do_test \
    --seed 422 > codeBERT-microsoft-codebert-base-novar-train_old_test_new-1weight_evaluation-Derived2.out  2>&1
if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-novar-train_old_test_new-1weight_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-novar-train_old_test_new-1weight_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./oldDataset/train.jsonl \
    --eval_data_file=./oldDataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/oldDataset/codeBERT/microsoft-codebert-base/novar-20weight \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base \
    --cache_dir=./cache/codeBERT \
    --loss_weight 20 \
    --do_test \
    --seed 422 > codeBERT-microsoft-codebert-base-novar-train_old_test_new-20weight_evaluation-Derived2.out  2>&1
if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-novar-train_old_test_new-20weight_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-novar-train_old_test_new-20weight_evaluation-Derived2 executed failed!"
#    exit 1
fi

echo "-----------------------------------------------------------------------------------------------------------------"
echo "Train dataset and test dataset"
echo "-----------------------------------------------------------------------------------------------------------------"

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/codeBERT/microsoft-codebert-base/novar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base \
    --cache_dir=./cache/codeBERT \
    --loss_weight 1 \
    --do_test \
    --seed 422 > codeBERT-microsoft-codebert-base-novar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1
if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/codeBERT/microsoft-codebert-base/novar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base \
    --cache_dir=./cache/codeBERT \
    --loss_weight 3 \
    --do_test \
    --seed 422 > codeBERT-microsoft-codebert-base-novar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1
if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/codeBERT/microsoft-codebert-base/usevar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base \
    --cache_dir=./cache/codeBERT \
    --loss_weight 1 \
    --do_test \
    --use_var \
    --seed 422 > codeBERT-microsoft-codebert-base-usevar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1
if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/codeBERT/microsoft-codebert-base/usevar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base \
    --cache_dir=./cache/codeBERT \
    --loss_weight 3 \
    --do_test \
    --use_var \
    --seed 422 > codeBERT-microsoft-codebert-base-usevar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/codeBERT/microsoft-codebert-base-mlm/novar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base-mlm \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base-mlm \
    --cache_dir=./cache/codeBERT \
    --loss_weight 1 \
    --do_test \
    --seed 422 > codeBERT-microsoft-codebert-base-mlm-novar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1
if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-mlm-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-mlm-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/codeBERT/microsoft-codebert-base-mlm/novar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base-mlm \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base-mlm \
    --cache_dir=./cache/codeBERT \
    --loss_weight 3 \
    --do_test \
    --seed 422 > codeBERT-microsoft-codebert-base-mlm-novar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-mlm-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-mlm-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi


python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/codeBERT/microsoft-codebert-base-mlm/usevar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base-mlm \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base-mlm \
    --cache_dir=./cache/codeBERT \
    --loss_weight 1 \
    --do_test \
    --use_var \
    --seed 422 > codeBERT-microsoft-codebert-base-mlm-usevar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-mlm-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-mlm-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/codeBERT/microsoft-codebert-base-mlm/usevar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/codeBERT/microsoft-codebert-base-mlm \
    --model_name_or_path=./pre-model/codeBERT/microsoft-codebert-base-mlm \
    --cache_dir=./cache/codeBERT \
    --loss_weight 3 \
    --do_test \
    --use_var \
    --seed 422 > codeBERT-microsoft-codebert-base-mlm-usevar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "codeBERT-microsoft-codebert-base-mlm-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "codeBERT-microsoft-codebert-base-mlm-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

echo "-----------------------------------------------------------------------------------------------------------------"

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/neulab/codebert-c/novar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/neulab/codebert-c \
    --model_name_or_path=./pre-model/neulab/codebert-c \
    --cache_dir=./cache/neulab \
    --loss_weight 1 \
    --do_test \
    --seed 422 > neulab-codebert-c-novar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "neulab-codebert-c-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "neulab-codebert-c-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/neulab/codebert-c/novar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/neulab/codebert-c \
    --model_name_or_path=./pre-model/neulab/codebert-c \
    --cache_dir=./cache/neulab \
    --loss_weight 3 \
    --do_test \
    --seed 422 > neulab-codebert-c-novar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "neulab-codebert-c-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "neulab-codebert-c-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/neulab/codebert-c/usevar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/neulab/codebert-c \
    --model_name_or_path=./pre-model/neulab/codebert-c \
    --cache_dir=./cache/neulab \
    --loss_weight 1 \
    --do_test \
    --use_var \
    --seed 422 > neulab-codebert-c-usevar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "neulab-codebert-c-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "neulab-codebert-c-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/neulab/codebert-c/usevar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/neulab/codebert-c \
    --model_name_or_path=./pre-model/neulab/codebert-c \
    --cache_dir=./cache/neulab \
    --loss_weight 3 \
    --do_test \
    --use_var \
    --seed 422 > neulab-codebert-c-usevar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "neulab-codebert-c-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "neulab-codebert-c-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/neulab/codebert-cpp/novar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/neulab/codebert-cpp \
    --model_name_or_path=./pre-model/neulab/codebert-cpp \
    --cache_dir=./cache/neulab \
    --loss_weight 1 \
    --do_test \
    --seed 422 > neulab-codebert-cpp-novar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "neulab-codebert-cpp-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "neulab-codebert-cpp-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/neulab/codebert-cpp/novar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/neulab/codebert-cpp \
    --model_name_or_path=./pre-model/neulab/codebert-cpp \
    --cache_dir=./cache/neulab \
    --loss_weight 3 \
    --do_test \
    --seed 422 > neulab-codebert-cpp-novar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "neulab-codebert-cpp-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "neulab-codebert-cpp-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/neulab/codebert-cpp/usevar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/neulab/codebert-cpp \
    --model_name_or_path=./pre-model/neulab/codebert-cpp \
    --cache_dir=./cache/neulab \
    --loss_weight 1 \
    --do_test \
    --use_var \
    --seed 422 > neulab-codebert-cpp-usevar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "neulab-codebert-cpp-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "neulab-codebert-cpp-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/neulab/codebert-cpp/usevar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/neulab/codebert-cpp \
    --model_name_or_path=./pre-model/neulab/codebert-cpp \
    --cache_dir=./cache/neulab \
    --loss_weight 3 \
    --do_test \
    --use_var \
    --seed 422 > neulab-codebert-cpp-usevar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "neulab-codebert-cpp-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "neulab-codebert-cpp-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi


python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/baai/bge-base-en-v1-5/novar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/baai/bge-base-en-v1-5 \
    --model_name_or_path=./pre-model/baai/bge-base-en-v1-5 \
    --cache_dir=./cache/baai \
    --loss_weight 1 \
    --do_test \
    --seed 422 > baai-bge-base-en-v1-5--novar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "baai-bge-base-en-v1-5--novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "baai-bge-base-en-v1-5--novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/baai/bge-base-en-v1-5/novar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/baai/bge-base-en-v1-5 \
    --model_name_or_path=./pre-model/baai/bge-base-en-v1-5 \
    --cache_dir=./cache/baai \
    --loss_weight 3 \
    --do_test \
    --seed 422 > baai-bge-base-en-v1-5--novar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "baai-bge-base-en-v1-5--novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "baai-bge-base-en-v1-5--novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/baai/bge-base-en-v1-5/usevar-1weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/baai/bge-base-en-v1-5 \
    --model_name_or_path=./pre-model/baai/bge-base-en-v1-5 \
    --cache_dir=./cache/baai \
    --loss_weight 1 \
    --do_test \
    --use_var \
    --seed 422 > baai-bge-base-en-v1-5--usevar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "baai-bge-base-en-v1-5--usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "baai-bge-base-en-v1-5--usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/baai/bge-base-en-v1-5/usevar-3weight-lr1e-5 \
    --model_type=auto \
    --tokenizer_name=./pre-model/baai/bge-base-en-v1-5 \
    --model_name_or_path=./pre-model/baai/bge-base-en-v1-5 \
    --cache_dir=./cache/baai \
    --loss_weight 3 \
    --do_test \
    --use_var \
    --seed 422 > baai-bge-base-en-v1-5--usevar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "baai-bge-base-en-v1-5--usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "baai-bge-base-en-v1-5--usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi


echo "-----------------------------------------------------------------------------------------------------------------"

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/salesforce/codet5-base/novar-1weight-lr1e-5 \
    --model_type=t5 \
    --tokenizer_name=./pre-model/salesforce/codet5-base \
    --model_name_or_path=./pre-model/salesforce/codet5-base \
    --cache_dir=./cache/salesforce \
    --loss_weight 1 \
    --do_test \
    --seed 422 > salesforce-codet5-base-novar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "salesforce-codet5-base-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "salesforce-codet5-base-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/salesforce/codet5-base/novar-3weight-lr1e-5 \
    --model_type=t5 \
    --tokenizer_name=./pre-model/salesforce/codet5-base \
    --model_name_or_path=./pre-model/salesforce/codet5-base \
    --cache_dir=./cache/salesforce \
    --loss_weight 3 \
    --do_test \
    --seed 422 > salesforce-codet5-base-novar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "salesforce-codet5-base-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "salesforce-codet5-base-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/salesforce/codet5-base/usevar-1weight-lr1e-5 \
    --model_type=t5 \
    --tokenizer_name=./pre-model/salesforce/codet5-base \
    --model_name_or_path=./pre-model/salesforce/codet5-base \
    --cache_dir=./cache/salesforce \
    --loss_weight 1 \
    --do_test \
    --use_var \
    --seed 422 > salesforce-codet5-base-usevar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "salesforce-codet5-base-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "salesforce-codet5-base-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/salesforce/codet5-base/usevar-3weight-lr1e-5 \
    --model_type=t5 \
    --tokenizer_name=./pre-model/salesforce/codet5-base \
    --model_name_or_path=./pre-model/salesforce/codet5-base \
    --cache_dir=./cache/salesforce \
    --loss_weight 3 \
    --do_test \
    --use_var \
    --seed 422 > salesforce-codet5-base-usevar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "salesforce-codet5-base-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "salesforce-codet5-base-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/salesforce/codet5-small/novar-1weight-lr1e-5 \
    --model_type=t5 \
    --tokenizer_name=./pre-model/salesforce/codet5-small \
    --model_name_or_path=./pre-model/salesforce/codet5-small \
    --cache_dir=./cache/salesforce \
    --loss_weight 1 \
    --do_test \
    --seed 422 > salesforce-codet5-small-novar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "salesforce-codet5-small-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "salesforce-codet5-small-novar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/salesforce/codet5-small/novar-3weight-lr1e-5 \
    --model_type=t5 \
    --tokenizer_name=./pre-model/salesforce/codet5-small \
    --model_name_or_path=./pre-model/salesforce/codet5-small \
    --cache_dir=./cache/salesforce \
    --loss_weight 3 \
    --do_test \
    --seed 422 > salesforce-codet5-small-novar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "salesforce-codet5-small-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "salesforce-codet5-small-novar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/salesforce/codet5-small/usevar-1weight-lr1e-5 \
    --model_type=t5 \
    --tokenizer_name=./pre-model/salesforce/codet5-small \
    --model_name_or_path=./pre-model/salesforce/codet5-small \
    --cache_dir=./cache/salesforce \
    --loss_weight 1 \
    --do_test \
    --use_var \
    --seed 422 > salesforce-codet5-small-usevar-train_test-1weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "salesforce-codet5-small-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "salesforce-codet5-small-usevar-train_test-1weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

python run.py \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fine_tuning \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --output_dir=./saved_models/Derived2/salesforce/codet5-small/usevar-3weight-lr1e-5 \
    --model_type=t5 \
    --tokenizer_name=./pre-model/salesforce/codet5-small \
    --model_name_or_path=./pre-model/salesforce/codet5-small \
    --cache_dir=./cache/salesforce \
    --loss_weight 3 \
    --do_test \
    --use_var \
    --seed 422 > salesforce-codet5-small-usevar-train_test-3weight-lr1e-5_evaluation-Derived2.out  2>&1

if [ $? -eq 0 ]; then
    echo "salesforce-codet5-small-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed successfully!"
else
    echo "salesforce-codet5-small-usevar-train_test-3weight-lr1e-5_evaluation-Derived2 executed failed!"
#    exit 1
fi

conda deactivate