# RWCVDF---Real-World Code Vulnerability Detection
Framework: From Data Preprocessing to
Multi-Feature Fusion Detection
## Dataset

Environment configuration file (docker image, requirements.txt, and freeze-conda-env.yml):





## Code of RWCVDF

### 文件结构

文件结构如下：

codebugmodel/

│

├── configs/

│	├── **config.yaml**

│	└── parse_args.py

├── dataset/

│	├── **DiverseVul-Derived**

├── joern/

│	├── data/

│	│	├── cpPickle.py

│	│	├──**dataEnhance.py**

│	│	├──**simplifyMiddleStructureCode.py**

│	│	├──statisticsCweProjectVul.py

│	│	└── validDataUsable.py

│	├── **joern_parse.py**

│	├── sensiAPI.txt

│	└── symbolizer.py

├── MultiFCode/

│	├── data/

│	│	├── processed/

│	│	│	└── **pre_embed/**

│	│	└── word2vec/

│	├── embedding/

│	│	├── embedding.py

│	│	└── vocabulary.py

│	├── **LanguageModel/**

│	│	├── cache/

│	│	├── dataset/

│	│	├── oldDataset/

│	│	├── pre-model/

│	│	├── saved_models/

│	│	├── **base_test.sh**

│	│	├── **evaluation_test_set.sh**

│	│	├── model.py

│	│	├── **run.py**

│	│	├── sensiAPI.txt

│	│	└── symbolizer.py

│	├── model/

│	├── wandb/

│	├── **dataSet.py**

│	├── **model.py**

│	├── sensiAPI.txt

│	├── tokenize_code.py

│	└── **train.py**

├── util/

│	└──utils.py

└── **Main.sh**



### config

config文件夹下包含了相应的训练配置信息

### dataset

dataset文件夹下为原始的数据，未生成相应的中间结构。

### joern

joern为生成中间结构的文件夹，运行joern_parse并给定相应参数后，会在data文件夹下生成相应的中间结构代码。

data文件夹下的代码为对中间结果进行处理的代码，simplifyMiddleStructureCode.py为中间结构简化代码，dataEnhance.py为数据增强代码，cpPickle.py，statisticsCweProjectVul.py，validDataUsable.py为工具代码。

### MultiFCode

MultiFCode为多特征融合代码，运行train.py则可以进行训练、验证和测试。其中model.py为模型结构代码。dataSet.py为数据加载代码。

data文件夹下存储预处理后的代码，processed中存储经过预处理但没有进行预嵌入的数据，processed/pre_embed中存储经过预处理和预嵌入的数据。word2vec中存储经过预训练的word2vec模型（用于预研，本文章中不使用）。

embedding文件夹下存储相应的word2vec+RNN模型，本研究中不使用。

LanguageModel文件夹下存储语言模型相关的内容，cache为缓存文件夹，dataset中存放用于训练、验证和测试的数据文件，pre-model中存储从huggingface中拉取的经过大规模预训练的语言模型。saved_models中存放在训练集上微调的模型。base_test.sh训练、验证测试脚本。evaluation_test_set.sh为仅测试脚本。model.py为模型代码。run.py为训练、验证、测试代码。

model文件夹存储运行train.py的模型输出。wandb文件夹存储相应的记录文件。

### util

util为工具文件夹，其下包含utils.py文件。该文件中包含了框架中用到的工具代码。

### Main.sh

Main.sh为总运行脚本，在完成环境配置、原始数据准备、预训练模型准备工作后，直接运行Main.sh即可运行全流程。若只执行部分步骤，则需要对该脚本进行相应的调整。



### Docker

docker中的conda环境如图：

![image-20250215135739449](./assets/image-20250215135739449.png)

joern环境为使用joern生成代码中间结构的环境，bugdetect环境为多表征融合检测的环境。



我们提供相应的docker镜像，存储在xxxxxx中，叫做zjx-cvd-v1.tar

使用命令``docker load < zjx-cvd-v1.tar ``可以加载该docker镜像。

使用下述命令运行容器：

``docker run -itd --name name(e.g. codebug) --shm-size 120g -p Port mapping (e.g. 48422:422) --runtime=nvidia -v Path mapping (e.g. /xxx/codebug:/workspace/codebug) zjx/cvd:v1 /bin/bash``

使用下述命令进入容器：

docker exec -it name (e.g. codebug) /bin/bash
