import os
import re
import time
from datetime import datetime

from typing import List

from datasets import load_dataset

from utils import (
    save_jsonl,
    llm_complete, llm_output_text, process_answer,\
    basic_stats,\
    question_prompt)

from prompts import COT_PROMPT

import argparse

_parser = argparse.ArgumentParser(
    description="Run inference on SimpleToM"
)
_parser.add_argument("--models", nargs='*', type=str, required=True, help="Generation model(s)")
_parser.add_argument(
    "--subset", type=str, required=False, default="all", help="Subset to run (valid subset: mental-state-qa, behavior-qa, judgment-qa, all)"
)
_parser.add_argument(
    "--limit", type=int, required=False, default=None, help="Number of instances to run (if not running full set, applied to each subset)"
)
_parser.add_argument(
    "--use-cot",
    type=lambda x: x.lower() == "true",
    default=False,
    help="Whether to ask the model to 'Think step by step' (True/False)"
)


# SUBSETS OF SimpleToM
DATASET_SUBSETS = ["mental-state-qa", "behavior-qa", "judgment-qa"]

# OUTPUT PATH
output_dir: str = os.path.abspath(os.path.join(os.getcwd(), "sample_outputs/inference_outputs"))



def run_qa(models: List[str], prompts_filtered, file_suffix, use_cot=False, classify_prompt=None):
    for model in models:
        file_name = os.path.join(output_dir, f"simpletom_stories_{model.replace('/', '--')}{file_suffix}.jsonl")
        if os.path.exists(file_name):
            raise ValueError(f"File {file_name} already exists!")

    print(f"Running SimpleToM on {len(prompts_filtered)} prompts and {len(models)} model(s)")
    for model in models[:]:
        all_res = []
        print(f"MODEL = {model}")
        p_counter = 0
        for prompt in prompts_filtered[:]:
            if p_counter % 50 == 0:
                print(f"{datetime.now()} {p_counter}/{len(prompts_filtered)}")
            p_counter += 1
            llm_output = None
            counter = 10
            max_tokens = 500 if use_cot else 20
            while llm_output is None and counter > 0:
                try:
                    llm_output = llm_complete([{"role": "system", "content": "You are a helpful assistant."},
                                               {"role": "user", "content": prompt['prompt']}], model,
                                              {'max_tokens': max_tokens, 'temperature': 0.0, "logprobs": True})
                except Exception as e:
                    print("LLM error. Retrying in 10 seconds...")
                    print(e)
                    llm_output = None
                    time.sleep(10)
                    counter -= 1
            prompt['llm_output'] = llm_output.model_dump()
            all_res.append(prompt)
        file_name = os.path.join(output_dir, f"simpletom_stories_{model.replace('/', '--')}{file_suffix}.jsonl")
        if classify_prompt is None:
            # process answers
            for x in all_res:
                answer_prefix = ".*the answer is" if use_cot else "answer"
                x.update(process_answer(x, answer_prefix=answer_prefix))
        else:
            for x in all_res:
                extracted_number = re.findall("\\d+", llm_output_text(x["llm_output"]))
                if extracted_number:
                    x["output_number"] = int(extracted_number[0])
                else:
                    x["output_number"] = None
        print(save_jsonl(file_name, all_res))
        if classify_prompt is None:
            print(basic_stats(all_res))


def main():
    # parse command line arguments
    args = _parser.parse_args()
    print(f"model(s) = {args.models}")
    print(f"subset = {args.subset}")
    print(f"num instances limit (per subset) = {args.limit}")
    print(f"use cot = {args.use_cot}")

    # load dataset
    # all subsets
    if args.subset == "all":
        ds_test_list_to_use = []
        for subset in DATASET_SUBSETS:
            ds = load_dataset("allenai/SimpleToM", subset)
            ds_test_list = ds['test'].to_list()  # SimpleToM is designed for testing models only, not for training
            if args.limit:
                ds_test_list_to_use += ds_test_list[:args.limit]
            else:
                ds_test_list_to_use += ds_test_list
    else:
        # single subset
        ds = load_dataset("allenai/SimpleToM", args.subset)
        ds_test_list_to_use = ds['test'].to_list() # SimpleToM is designed for testing models only, not for training
        if args.limit:
            ds_test_list_to_use = ds_test_list_to_use[:args.limit]
    print(f"Using {len(ds_test_list_to_use)} instances from dataset")

    # prepare prompt
    prompts = []
    add_either_option = False
    add_neither_option = False
    use_cot = args.use_cot
    for story_data in ds_test_list_to_use:
        if use_cot:
            prompt = question_prompt(story_data, add_either_option=add_either_option,
                                     add_neither_option=add_neither_option, final_prompt=COT_PROMPT)
        else:
            prompt = question_prompt(story_data, add_either_option=add_either_option,
                                 add_neither_option=add_neither_option)
        prompt['id'] = story_data['id']
        prompts.append(prompt)

    print("-----Example prompts-----")
    for p in prompts[:3]:
        print(p['prompt'] + "\n-----\n")

    # run model(s) and save output
    run_qa(args.models, prompts,f"_subset_{args.subset}_limit_{args.limit}_cot_{use_cot}".lower(), use_cot=use_cot)





if __name__ == "__main__":
    main()
