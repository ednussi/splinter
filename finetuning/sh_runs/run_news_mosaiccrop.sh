#!/bin/sh
#SBATCH --time=4-0:0:0
#SBATCH --gres=gpu:1
source  /cs/labs/gabis/ednussi/v1/bin/activate
export MODEL="roberta-base"
export BASEPATH="/cs/labs/gabis/ednussi/splinter/finetuning"

d='newsqa'
for i in 256
do
  for j in 42 43 44 45 46
  do
    echo "Loop $i-$j"
    EXPNAME="$d-mosaiccrop"
    TRAINFILE="$BASEPATH/mrqa_data/$d/$d-train-seed-$j-num-examples-$i.jsonl"
    TESTFILE="$BASEPATH/mrqa_data/$d/dev.jsonl"
    OUTPUTDIR="$BASEPATH/results/$EXPNAME/output-$i-$j"
    AUGMENTATION="mosaiccrop-2-False"
    mkdir -p -- $OUTPUTDIR
    python run_mrqa.py --model_type=$MODEL --model_name_or_path=$MODEL --qass_head=False --tokenizer_name=$MODEL --output_dir=$OUTPUTDIR --train_file=$TRAINFILE --predict_file=$TESTFILE --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=12 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=200 --num_train_epochs=10 --seed=$j --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir --aug $AUGMENTATION
    rm -rf "$OUTPUTDIR/pytorch_model.bin"
  done
done
