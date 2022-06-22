# 環境構築
```
sudo nvidia-docker run -it -d --name semsol \
             -v /raid:/raid \
             -v /share_6:/share_6 \
             -v /share_7:/share_7 \
              nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04 \
             /bin/bash

apt update
apt install -y git python3 python3-pip vim wget unzip
pip3 install --upgrade pip
pip3 install numpy==1.19.5 setproctitle==1.2.2 tqdm==4.56.2 transformers==2.8.0 nltk==3.6.7 spacy==3.2.3
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

apt install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
apt install -y language-pack-ja-base language-pack-ja ibus-mozc
echo 'export LANG=ja_JP.UTF-8' | tee -a ~/.bashrc
echo 'export LANGUAGE=ja_JP:ja' | tee -a ~/.bashrc

python3 -m spacy download zh_core_web_sm
python3 -m spacy download en_core_web_sm
```

# データ配置
```
wget -O DoubanConversaionCorpus.zip 'https://www.dropbox.com/s/90t0qtji9ow20ca/DoubanConversaionCorpus.zip?dl=0'
unzip DoubanConversaionCorpus.zip
head -n 50000 DoubanConversaionCorpus/dev.txt > DoubanConversaionCorpus/dev_1.txt
mv DoubanConversaionCorpus/dev_1.txt DoubanConversaionCorpus/dev.txt
mv DoubanConversaionCorpus/*.txt douban_data/
rm -r DoubanConversaionCorpus DoubanConversaionCorpus.zip
```

# 学習データ作成
```
python3 Data_processing.py --task douban
```

# Pre-training
```
CUDA_VISIBLE_DEVICES=1 python3 -u Pre-training/douban_final.py \
    --num_train_epochs 50 \
    --train_batch_size 96 \
    --use_semantic \
    --use_topic \
    --output_dir ./Pre-training/PT_checkpoint/douban
```

# Fine-tuning
```
CUDA_VISIBLE_DEVICES=0 python3 -u Fine-Tuning/Response_selection.py \
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

# Douban/sematicの精度再現
```
CUDA_VISIBLE_DEVICES=0 python3 -u Fine-Tuning/Response_selection.py \
    --task douban \
    --use_semantic \
    --load_checkpoint ./SemSol_without_utterances_best.pt \
    --score_file_path ./douban_load-best_scorefile.txt
```

# 処理時間
```
・Topic学習
	Post-training
		1epoch＝3:13:36
		1stepの処理時間≒0.4秒

	Fine-tuning
		1epoch＝1:08:25
		1stepの処理時間≒0.4秒

・semantic学習
	Post-training
		1epoch＝8:20:48
		1stepの処理時間≒1.01秒

	Fine-tuning
		1epoch＝5:08:44
		1stepの処理時間≒1.77秒

・topic＋semantic学習
	Post-training
		1epoch＝8:35:59
		1stepの処理時間≒1.01秒

	Fine-tuning
		1epoch＝5:05:15
		1stepの処理時間≒1.77秒
```

# doubanデータ
```
https://github.com/AIAquapolis/BERT_FP_SemSol/tree/feature/20220620_original
	https://www.dropbox.com/s/90t0qtji9ow20ca/DoubanConversaionCorpus.zip?dl=0
```
