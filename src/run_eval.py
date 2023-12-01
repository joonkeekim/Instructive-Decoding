import os
import json
import argparse
import wandb
import numpy as np
import random
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from compute_metrics import compute_metrics
from sni_generator import SniGenerator
from mmlu_generator import MMLUGenerator


def main(args):
    '''
    Evaluation for Instruction-tuned T5 over SuperNatInst Dataset.
    Description:
        1. Load test tasks
        2. For each task, load the task file and generate the outputs
        3. Compute the metrics
        4. Save the results and metrics
    '''
    # if "3b" not in args.group_name and "11b"  not in args.group_name:
        # raise NotImplementedError("Other models except 3b and 11b are not implemented")

    # wandb initialization
    wandb.init(project="superNI project", 
               entity="Instruct-decode",
               group=args.wandb_group,
               name=args.wandb_name,
               config=args)

    # seed initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Tk-Instruct 3b or 11b
    if not args.is_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, resume_download=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        model = LlamaForCausalLM.from_pretrained(args.model_name, resume_download=True)
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)

    if args.task == "super_ni":
        generator = SniGenerator(model, tokenizer, args)
        generator.super_ni_eval()
    elif args.task == "mmlu":
        generator = MMLUGenerator(model, tokenizer, args)
        generator.mmlu_eval()
            
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Process Configs")
    parser.add_argument("--task", default="super_ni", hoices=["none", "mmlu"], type=str)
    parser.add_argument("--is_decoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model_name", default="allenai/tk-instruct-3b-def", type=str)
    parser.add_argument("--output_path", default="outputs/tk-3b-margi-pos", type=str)

    # super NI
    parser.add_argument("--test_tasks_path", default="./data/supni/splits/default/test_tasks.txt", type=str)
    parser.add_argument("--task_files_path", default="./data/supni/tasks", type=str)

    # device config
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", default=1, type=int)

    # wandb config
    parser.add_argument("--wandb_group", default="3b", type=str)
    parser.add_argument("--wandb_name", default="tk-3b-test", type=str)

    # instruction configs
    parser.add_argument("--answer_with", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noisy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logit_analysis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--neg_type", default="none", type=str, 
                        choices=["none", "trunc_shuf", "random_words", "opposite", # noisy variants
                                 "none_options", "opposite_options", "opposite_full"]) # noisy variants for hybrids
    parser.add_argument("--trunc_rate", default=0.6, type=float) # when you use 
    parser.add_argument("--num_rand_words", default=1, type=int)
    
    # decode config
    parser.add_argument("--eps", default=0, type=float)
    parser.add_argument("--dec_type", default="logit", type=str, choices=["logit", "prob"])
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)

    # sampling decoding
    parser.add_argument("--top_p", default=-1, type=float)
    parser.add_argument("--top_k", default=-1, type=float)
    parser.add_argument("--temp", default=-1, type=float)
    parser.add_argument("--candidate", action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()
    main(args)