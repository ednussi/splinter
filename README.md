## Run Locally

### Run Baseline 1 - file
python run_mrqa.py --model_type=roberta-base --model_name_or_path=roberta-base --qass_head=False --tokenizer_name=roberta-base --output_dir=outputs --train_file=squad/mosaic_unite_npairs-4/squad-train-seed-42-num-examples-16.jsonl --predict_file="squad/dev.jsonl" --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=16 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=1 --num_train_epochs=1 --seed=42 --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir

### LATEST RUN COMMAND LOCALLY
python run_mrqa.py --model_type=roberta-base --model_name_or_path=roberta-base --qass_head=False --tokenizer_name=roberta-base --output_dir=outputs --train_file=squad/mosaic_unite_npairs-4/squad-train-seed-42-num-examples-16.jsonl --predict_file="squad/dev.jsonl" --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=16 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=1 --num_train_epochs=1 --seed=42 --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir --aug context-shuffle

### Run Basic mosaic aug
python run_mrqa.py --model_type=roberta-base --model_name_or_path=roberta-base --qass_head=False --tokenizer_name=roberta-base --output_dir=outputs --train_file=squad/mosaic_unite_npairs-4/squad-train-seed-42-num-examples-16.jsonl --predict_file="squad/dev.jsonl" --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=16 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=1 --num_train_epochs=1 --seed=42 --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir --aug mosaic-2-True
python run_mrqa.py --model_type=roberta-base --model_name_or_path=roberta-base --qass_head=False --tokenizer_name=roberta-base --output_dir=outputs --train_file=squad/mosaic_unite_npairs-4/squad-train-seed-42-num-examples-16.jsonl --predict_file="squad/dev.jsonl" --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=16 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=1 --num_train_epochs=1 --seed=42 --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir --aug context-shuffle 

### Training Over Single File
`python run_mrqa.py --model_type=roberta-base --model_name_or_path=roberta-base --qass_head=False --tokenizer_name=roberta-base --output_dir=outputs --train_file=squad/mosaic_unite_npairs-4/squad-train-seed-42-num-examples-16.jsonl --predict_file="squad/dev.jsonl" --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=16 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=1 --num_train_epochs=1 --seed=42 --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir`

## Run Remote
### Single sbatch run
sbatch --gres=gpu:rtx2080:1 sh_runs/run_baseline.sh

### Run Remote on GPU SINGLE (SRUN)
python run_mrqa.py --model_type=roberta-base --model_name_or_path=roberta-base --qass_head=False --tokenizer_name=roberta-base --output_dir=results/single_run_test --train_file=squad/mosaic_unite_npairs-4/squad-train-seed-42-num-examples-16.jsonl --predict_file="squad/dev.jsonl" --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=16 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=1 --num_train_epochs=1 --seed=42 --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir --aug concat-coherent-text


## Work

1) `ssh ednussi%phoenix-gw@gw.cs.huji.ac.il`

2) `srun --pty --gres=gpu:rtx2080 bash`

3) `sbatch --time=2:0:0 --gres=gpu:rtx2080:1 sh_runs/run_baseline.sh`

4) Manually
```angular2html
cd /cs/labs/gabis/ednussi/
source venv_splinter/bin/activate
cd splinter/finetuning

export MODEL="roberta-base"
export OUTPUT_DIR="output" 
python run_mrqa.py  --model_type=roberta-base  --model_name_or_path=$MODEL  --qass_head=False  --tokenizer_name=$MODEL  --output_dir=$OUTPUT_DIR  --train_file="../squad/squad-train-seed-42-num-examples-16.jsonl"  --predict_file="../squad/dev.jsonl"  --do_train  --do_eval  --cache_dir=.cache  --max_seq_length=384  --doc_stride=128  --threads=4  --save_steps=50000  --per_gpu_train_batch_size=12  --per_gpu_eval_batch_size=16  --learning_rate=3e-5  --max_answer_length=10  --warmup_ratio=0.1  --min_steps=200  --num_train_epochs=10  --seed=42  --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir
```
# Splinter

This repository was forked on 25th April 2021, from the original splinter repo matching the "[Few-Shot Question Answering by Pretraining Span Selection](https://arxiv.org/abs/2101.00438)" Paper.

## Step-by-Step Install
Maybe needed for `spacy`
`export BLIS_REALLY_COMPILE=1`


## Step-by-Step Reproducing Old Roberta-base Results
### Create new virtual env
```angular2html
conda create -n splinter-env python=3.8
conda activate splinter-env
```

### clone and install requirements
```angular2html
git clone git@github.com:ednussi/splinter.git
cd splinter/finetuning
pip install -r requirements.txt
``` 

### Donwload Original Paper Data
```angular2html
mkdir mrqa_data
cd mrqa_data
curl -L https://www.dropbox.com/sh/pfg8j6yfpjltwdx/AAC8Oky0w8ZS-S3S5zSSAuQma?dl=1 > mrqa-few-shot.zip
unzip mrqa-few-shot.zip
cd ..
```

### Run & Verify correctness so far
Note on single Titan X took ~45 minutes to train + ~1 hour to eval.
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
