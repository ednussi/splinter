# Splinter

This repository was forked on 25th April 2021, from the original splinter repo matching the "[Few-Shot Question Answering by Pretraining Span Selection](https://arxiv.org/abs/2101.00438)" Paper.

## Step-by-Step Reproducing Roberta-base Results
### Create new virtual env
```angular2html
conda create -n splinter-env python=3.8
conda cativate splinter-env
```

### clone and install requirements
```angular2html
git clone git@github.com:ednussi/splinter.git
cd splinter/finetuning
pip install -r requirements.txt
``` 

### Donwload Original Paper Data
```angular2html
curl -L https://www.dropbox.com/sh/pfg8j6yfpjltwdx/AAC8Oky0w8ZS-S3S5zSSAuQma?dl=1 > mrqa-few-shot.zip
unzip mrqa-few-shot.zip
```

### Run & Verify correctness so far
Note on single Titan X took ~45 minutes to run.
```angular2html
mkdir outputs
python run_mrqa.py \
    --model_type=roberta-base \
    --model_name_or_path=roberta-base \
    --qass_head=False \
    --tokenizer_name=roberta-base \
    --output_dir="outputs/output32-42-test" \
    --train_file="squad/squad-train-seed-42-num-examples-32.jsonl" \
    --predict_file="squad/dev.jsonl" \
    --do_train \
    --do_eval \
    --cache_dir=.cache \
    --max_seq_length=384 \
    --doc_stride=128 \
    --threads=4 \
    --save_steps=50000 \
    --per_gpu_train_batch_size=12 \
    --per_gpu_eval_batch_size=12 \
    --learning_rate=3e-5 \
    --max_answer_length=10 \
    --warmup_ratio=0.1 \
    --min_steps=200 \
    --num_train_epochs=10 \
    --seed=42 \
    --use_cache=False \
    --evaluate_every_epoch=False
```
### To reproduce results run this .sh script
```angular2html
export MODEL="roberta-base"
for i in 64 128 256
do
  for j in 42 43 44 45 46
  do
    echo "Loop $i-$j"
    python run_mrqa.py --model_type=$MODEL --model_name_or_path=$MODEL --qass_head=False --tokenizer_name=$MODEL --output_dir="outputs/output$i-$j" --train_file="squad/squad-train-seed-$j-num-examples-$i.jsonl" --predict_file="squad/dev.jsonl" --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=12 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=200 --num_train_epochs=10 --seed=$j --use_cache=False --evaluate_every_epoch=False
```
### Results
![image](https://user-images.githubusercontent.com/10045688/116010500-26f25e80-a5d4-11eb-9677-34c120e52d81.png)
[Original ticke](https://github.com/oriram/splinter/issues/1#issuecomment-823697203)
