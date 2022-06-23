## Douban Conversation Corpus
|  Model  |  MAP  |  MRR  |  P@1  |  R1  |  R2  |  R5  |Paper and Code|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|SemSol(W/O utterances)|_**0.651**_|_**0.687**_|0.510|0.328|0.552|0.877|Response Selection utilizing Semantics underlies Multi-Turn Open-Domain Conversations (under review)|
|SemSol|0.640|0.678|0.511|_**0.330**_|0.520|0.870|Response Selection utilizing Semantics underlies Multi-Turn Open-Domain Conversations (under review)|
|BERT-FP (Han et al., 2021)|0.644|0.680|0.512|0.324|0.542|0.870|Fine-grained Post-training for Improving Retrieval-based Dialogue Systems. NAACL 2021.|
|SA-BERT+HCL (Su et al., 2021)|0.639|0.681|0.514|0.330|0.531|0.858|Dialogue Response Selection with Hierarchical Curriculum Learning. ACL 2021.|
|UMS_BERT+ (Whang et al., 2020)|0.625|0.664|0.499|0.318|0.482|0.858|Do Response Selection Models Really Know What’s Next? Utterance Manipulation Strategies for Multi-turn Response Selection. AAAI 2021.|
|SA-BERT (Gu et al., 2020)|0.619|0.659|0.496|0.313|0.481|0.847|Speaker-Aware BERT for Multi-Turn Response Selection in Retrieval-Based Chatbots. CIKM 2020.|
|DCM (Li et al., 2020)|0.611|0.649|-|0.294|0.498|0.842|Deep context modeling for multi-turn response selection in dialogue systems. Information Processing & Management 2020.|

## Setup and Dependencies
&nbsp;This code is implemented using PyTorch v1.8.2 version and NLTK v3.6.7
version and provides out-of-the-box support with CUDA 11.2. Anaconda is
recommended for setting up this codebase.

An example of running our codebase is as follows:
```
nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04
# apt update
# apt install -y git python3 python3-pip vim wget unzip
# pip3 install --upgrade pip
# pip3 install numpy==1.19.5 setproctitle==1.2.2 tqdm==4.56.2 transformers==2.8.0 nltk==3.6.7 spacy==3.2.3
# pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
# python3 -m spacy download zh_core_web_sm
# python3 -m spacy download en_core_web_sm
```

## Source code of SemSol
&nbsp;Source code of semsol is included in “Pre-Training” and “Fine-tuning”directories. Our code is built based on the implementation of [BERT-FP](https://github.com/hanjanghoon/BERT_FP)
&nbsp;The core component for utterance embeddings and that for semantic embeddings are implemented in “semsol_model.py” and “fine_semsol_model.py”, respectively. 
This code is reimplemented as a fork of huggingface/transformers.

## Douban dataset and our checkpoint
&nbsp;Douban dataset can be aquired from the github site of original Douban paper.  
&nbsp;We provide the checkpoint for SemSol (w/o utterances) for Douban dataset learned by our evaluation. The checkpoint, “SemSol_without_utterances_best.pt”, can be downloaded from the below and should be put in “SemSol” directory.  
&nbsp;The result by this checkpoint can be aquired by the following inference command:  
  
- [SemSol_without_utterances_best.pt](https://www.dropbox.com/s/9r36z81iu940pd5/SemSol_without_utterances_best.zip?dl=0)

```
# CUDA_VISIBLE_DEVICES=0 python3 -u Fine-Tuning/Response_selection.py \
      --task douban \
      --use_semantic \ 
      --load_checkpoint ./SemSol_without_utterances_best.pt \
      --score_file_path ./douban_load-best_scorefile.txt
```

## Pre-training
&nbsp;The example for pre-training SemSol is as follows:  
1. Put Douban dataset in “./douban_data” directory.
```
# wget -O DoubanConversaionCorpus.zip 'https://www.dropbox.com/s/90t0qtji9ow20ca/DoubanConversaionCorpus.zip?dl=0'
# unzip DoubanConversaionCorpus.zip
# head -n 50000 DoubanConversaionCorpus/dev.txt > DoubanConversaionCorpus/dev_1.txt
# mv DoubanConversaionCorpus/dev_1.txt DoubanConversaionCorpus/dev.txt
# mv DoubanConversaionCorpus/*.txt douban_data/
# rm -r DoubanConversaionCorpus DoubanConversaionCorpus.zip
```

2. Create training data as:
```
# python3 Data_processing.py --task douban
```

3. Run pre-training as:
```
# CUDA_VISIBLE_DEVICES=1 python3 -u Pre-training/douban_final.py \
    --num_train_epochs 50 \
    --train_batch_size 96 \
    --use_semantic \
    --use_topic \
    --output_dir ./Pre-training/PT_checkpoint/douban
```

## Fine-tuning
&nbsp;The example for fine-tuning SemSol is as follows:
```
# CUDA_VISIBLE_DEVICES=0 python3 -u Fine-Tuning/Response_selection.py \
    --task douban \
    --use_semantic \
    --use_topic \
    --is_training \
    --save_path ./Fine-Tuning/FT_checkpoint/douban_fine \
    --load_checkpoint ./Pre-training/PT_checkpoint/douban/checkpoint2/bert.pt \
    --batch_size 96 \
    --epochs 2 \
    --score_file_path ./Fine-Tuning/FT_checkpoint/douban_fine_scorefile.txt \
    --learning_rate 3e-5
```
