import argparse
from tqdm import tqdm
import re
import os

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exps, -e",
        type=str,
        nargs='+',
        required=True,
        help="Expirement path to results",
    )
    return parser.parse_args()

def get_f1_em_dict(exp_paths):
    base_path = os.getcwd()
    for exp in exp_paths:

        res_dict = {}
        for num_examples in tqdm([16, 32, 64, 128, 256], desc='Base Examples'):

            for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):

                res_folder_path = f'{exp}/output-{num_examples}-{seed}'
                res_file = f'{base_path}/{res_folder_path}/eval_results.txt'
                if os.path.exists(res_file):
                    with open(res_file, "r") as f:
                        lines = f.readlines()

                    f1 = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_f1 = ')][0])
                    em = re.findall("\d+\.\d+", [x for x in lines if x.startswith('best_exact = ')][0])
                    res_dict[f'{num_examples}-{seed}'] = {'exact':em, 'f1':f1}

        # plot this aug
        print(f'============ {exp} ============')
        print(res_dict)

if __name__ == '__main__':
    args = init_parser()
    get_f1_em_dict(args.exps)