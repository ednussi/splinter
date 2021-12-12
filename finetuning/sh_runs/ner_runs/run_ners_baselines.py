import os
from tqdm import tqdm
if __name__ == '__main__':
    exp_name = 'baseline'
    base_cmd = 'python /d/Thesis/splinter/finetuning/run_ner.py --model_name_or_path bert-base-uncased --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10'
    for dataset_name in tqdm(['conll2003', 'wnut_17', 'ncbi_disease', 'species_800', 'bc2gm_corpus'], desc='datasets'):
        for num_samples in tqdm([1024, 512, 256, 128, 64, 32, 16], desc='examples'):
            for seed in [42, 43, 44, 45, 46]:
                output_dir = f'ner_res/{dataset_name}-baseline/{num_samples}-{seed}'
                cmd = f'{base_cmd} --dataset_name {dataset_name} --seed {seed} --max_train_samples {num_samples} --output_dir {output_dir}'
                print(f'Running {cmd}')
                os.system(cmd)
                remove_cmd = f"rm -rf {dataset_name}-baseline/{num_samples}-{seed}/pytorch_model.bin"
                os.system(remove_cmd)
    print('Done')