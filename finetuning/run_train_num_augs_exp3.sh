#!/bin/sh
#SBATCH --time=2-0:0:0
#SBATCH --gres=gpu:rtx2080:1
source  /cs/labs/gabis/ednussi/v1/bin/activate
export MODEL="roberta-base"
export AUG="delete-random"

for num_augs in 1 2 3
do
  for i in 256 128 64 32 16
  do
    for j in 42 43 44 45 46
    do
      echo "Loop $AUG-$i-$j"
      OUTPUTDIR="outputs/num_augs_exp/output-$AUG-$i-$j-$num_augs"
      mkdir -p -- $OUTPUTDIR
      python run_mrqa.py --model_type=$MODEL --model_name_or_path=$MODEL --qass_head=False --tokenizer_name=$MODEL --output_dir=$OUTPUTDIR --train_file="squad/num_augs_exp/$AUG-embed_$num_augs-count/squad-train-seed-$j-num-examples-$i-augs_$AUG-embed-$num_augs.jsonl" --predict_file="squad/dev.jsonl" --do_train --do_eval --cache_dir=.cache --max_seq_length=384 --doc_stride=128 --threads=4 --save_steps=50000 --per_gpu_train_batch_size=12 --per_gpu_eval_batch_size=12 --learning_rate=3e-5 --max_answer_length=10 --warmup_ratio=0.1 --min_steps=200 --num_train_epochs=10 --seed=$j --use_cache=False --evaluate_every_epoch=False --overwrite_output_dir
    done
  done
done
