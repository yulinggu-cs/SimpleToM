import os
import re
import time
from datetime import datetime

from story_generation.utils import process_output, make_prompt
from inference.utils import (
    load_jsonl, save_jsonl, \
    llm_complete, llm_output_text, \
    fill_template, parse_template )
from story_generation.prompts import SCENARIO_TEMPLATE, INSTRUCTIONS_ENTITY_BRAINSTORM, INSTRUCTIONS

from story_generation.seed_data import (SCENARIO_STORY_EXAMPLES, SCENARIO_NAMES_TO_STORY,\
                       SCENARIO_STORY_EXAMPLES_2, SCENARIO_NAMES_TO_STORY_2)



import argparse

_parser = argparse.ArgumentParser(
    description="Generate new SimpleToM stories"
)
_parser.add_argument("--entity-models", nargs='*', type=str, required=True, help="Generation model(s) for entity brainstorming")

_parser.add_argument("--story-models", nargs='*', type=str, required=True, help="Generation model(s) for writing the stories")

_parser.add_argument(
    "--story-seed-round", type=int, required=False, default=0, help="Indicate to use seed stories from round 1 or 2, uses seed stories from both rounds by default"
)

_parser.add_argument(
    "--exclude-seed-stories",
    type=lambda x: x.lower() == "true",
    default=False,
    help="Whether to exclude curated seed stories in output"
)

_parser.add_argument("--running-mode", type=str, required=False, default="all", help='Indicate "all" to run generation on all categories and "test" to run a small test to check if the pipeline is working first')

_parser.add_argument("--file-prefix", type=str, required=False, default="new", help='Desired file prefix for saving outputs')



output_dir: str = os.path.abspath(os.path.join(os.getcwd(), "sample_outputs/story_gen_outputs"))
entity_keyinfo_template = "ENTITIES:\n{entities}\n\nKEY INFORMATION:\n{key_information}"


def process_sample_story(examples_dict, story, scenario, identifier):
    # examples_dict can be SCENARIO_STORY_EXAMPLES or SCENARIO_STORY_EXAMPLES_2
    story_dict = examples_dict[story]
    output = {"story_example": story, "scenario": scenario, "scenario_text":story_dict['scenario'],
              "entities": None,
              "output": fill_template(SCENARIO_TEMPLATE, examples_dict[story])}
    return process_output(output, identifier)


def prepare_entity_brainstorming_prompts():

    prompts_brainstorm = []
    for scenario_name in SCENARIO_NAMES_TO_STORY:
        story = SCENARIO_STORY_EXAMPLES[SCENARIO_NAMES_TO_STORY[scenario_name]]
        story_entity_example = fill_template(entity_keyinfo_template, story)
        prompt = fill_template(INSTRUCTIONS_ENTITY_BRAINSTORM, {
            "scenario": story['scenario'],
            "scenario_entity_example": story_entity_example})
        prompts_brainstorm.append({'scenario_name': scenario_name,
                                 'scenario_text': story['scenario'],
                                 'example': story_entity_example,
                                 'prompt': prompt})

    return prompts_brainstorm

def run_models_to_brainstorm_entities(prompts_brainstorm, file_prefix, brainstorming_models=['gpt-4-0125-preview', 'claude-3-opus-20240229']):

    all_res = []
    for prompt in prompts_brainstorm[:]:
        for model in brainstorming_models[:]:
            print(f"{datetime.now()} {prompt['scenario_name']} {model}")
            llm_output = None
            counter = 10
            while llm_output is None and counter > 0:
                try:
                    llm_output = llm_complete([{"role": "system", "content": "You are a helpful, creative assistant."},
                                               {"role": "user", "content": prompt['prompt']}], model,
                                              {'max_tokens': 1000, 'temperature': 0.5})
                except Exception as e:
                    print("LLM error. Retrying in 10 seconds...")
                    print(e)
                    llm_output = None
                    time.sleep(10)
                    counter -= 1
            res = prompt.copy()
            res['model'] = model
            res['llm_output'] = llm_output.model_dump() # llm_output.dict() Deprecated in Pydantic V2.0
            all_res.append(res)
    print(f"{datetime.now()} ALL DONE!")

    entities_file = save_jsonl(os.path.join(output_dir, f"{file_prefix}_part1_entity_brainstorm_{'_'.join(brainstorming_models)}.jsonl"), all_res)
    print(f"SAVED brainstormed entities to {entities_file}")
    return entities_file

def match_seed_story_scenario_with_entities(scenario_names_2_seed_story, entities_brainstormed_res):
    story_and_scenarios_to_run = []
    for res in entities_brainstormed_res:
        seed_story = scenario_names_2_seed_story[res['scenario_name']]
        llm_text = llm_output_text(res['llm_output'])
        suggestions = [parse_template(entity_keyinfo_template, x) for x in re.split("Example \\d+:", llm_text) if
                       "ENTITIES" in x]
        for suggestion in suggestions:
            entities = suggestion['entities'].strip()
            story_and_scenarios_to_run.append({"seed_story": seed_story, "scenario_name": res['scenario_name'],
                                               "entities": entities, "entities_source": res['model']})
    return story_and_scenarios_to_run


def generate_story_components_raw(instruction_prompt, story_models, story_and_scenarios_to_run, verbose=False):
    stored_outputs = []
    prompt_gen = None

    llm_options = {"max_tokens": 2000, "temperature": 0.5, "n": 1}
    for story_and_scenario in story_and_scenarios_to_run[:]:
        for model in story_models:
            print(f"{datetime.now()} Processing {model} for {story_and_scenario}...")
            story = story_and_scenario['seed_story']
            scenario_name = story_and_scenario['scenario_name']
            if story in SCENARIO_STORY_EXAMPLES:
                story_data = SCENARIO_STORY_EXAMPLES[story]
                new_scenario = SCENARIO_STORY_EXAMPLES[SCENARIO_NAMES_TO_STORY[scenario_name]]['scenario']
            else:
                story_data = SCENARIO_STORY_EXAMPLES_2[story]
                new_scenario = SCENARIO_STORY_EXAMPLES_2[SCENARIO_NAMES_TO_STORY_2[scenario_name]]['scenario']
            new_entities = story_and_scenario['entities']
            prompt_gen = make_prompt(story_data, new_scenario,
                                     new_entities,
                                     instruction=instruction_prompt)
            llm_output = None
            counter = 10
            while llm_output is None and counter > 0:
                try:
                    llm_output = llm_complete(
                        [{"role": "system", "content": "You are a creative, but rigorous author."},
                         {"role": "user", "content": prompt_gen}], model, llm_options)
                except Exception as e:
                    print("LLM error. Retrying in 10 seconds...")
                    print(e)
                    llm_output = None
                    time.sleep(10)
                    counter -= 1
            stored_outputs.append({"story_example": story, "scenario": scenario_name,
                                   "scenario_text": new_scenario,
                                   "entities": new_entities, "output": llm_output.model_dump()})

    if verbose:
        print(prompt_gen)

    return stored_outputs

def process_generated_stories(raw_stories_generated, exclude_seed=False):
    stories_processed = []
    missed = []
    if not exclude_seed:
    # Include all original stories
        for scenario, story in SCENARIO_NAMES_TO_STORY.items():
            stories_processed += process_sample_story(SCENARIO_STORY_EXAMPLES, story, scenario, story)
        for scenario, story in SCENARIO_NAMES_TO_STORY_2.items():
            stories_processed += process_sample_story(SCENARIO_STORY_EXAMPLES_2, story, scenario, "seed_" + story)
    id_counter = 1
    for output1 in raw_stories_generated:
        current_id = f"user_new_gen{id_counter}"
        # print(current_id)
        id_counter += 1
        try:
            stories_processed += process_output(output1, current_id)
        except Exception:
            missed.append((current_id, id_counter - 2))
            print(f"SKIPPING {current_id} DUE TO FAILED PROCESSING")
    print(f"Processed {len(stories_processed)} questions up to id_counter = {id_counter - 1}")

    return stories_processed, missed


def main():
    # parse command line arguments
    args = _parser.parse_args()

    # Add "scenario" field which describes the information asymmetry for second round of seed stories
    for k, v in SCENARIO_STORY_EXAMPLES_2.items():
        v['scenario'] = SCENARIO_STORY_EXAMPLES[SCENARIO_NAMES_TO_STORY[v['scenario_name']]]['scenario']

    print(f"Entity brainstorming model(s) = {args.entity_models}")
    print(f"Story generation model(s) = {args.story_models}")
    print(f"Seed story round = {args.story_seed_round} (0 indicates use both available rounds)")
    print(f"Exclude seed stories = {args.exclude_seed_stories}")
    print(f"Running mode = {args.running_mode}")
    print(f"File prefix for saving outputs = {args.file_prefix}")

    running_mode = args.running_mode
    file_prefix = args.file_prefix

    ## Step 0: Examine example stories
    # print(list(SCENARIO_STORY_EXAMPLES.items())[0])
    # print(list(SCENARIO_STORY_EXAMPLES_2.items())[0])

    ## Step 1: Brainstorm entities
    # 1.1 Prepare prompts
    prompts_brainstorm = prepare_entity_brainstorming_prompts()
    print(prompts_brainstorm[-1]['prompt'])
    print(len(prompts_brainstorm))

    # 1.2 Run entity brainstorming prompt using desired model(s)
    if running_mode == "test":
        entities_file = run_models_to_brainstorm_entities(prompts_brainstorm[:2], file_prefix, args.entity_models)
    else:
        entities_file = run_models_to_brainstorm_entities(prompts_brainstorm, file_prefix, args.entity_models)
    entities_brainstormed_res = load_jsonl(entities_file)

    # 1.3 match desired seed story and scenario with brainstormed entities
    story_and_scenarios_to_run_all = []
    if args.story_seed_round == 1:
        story_and_scenarios_to_run_all = match_seed_story_scenario_with_entities(SCENARIO_NAMES_TO_STORY, entities_brainstormed_res)
    elif args.story_seed_round == 2:
        story_and_scenarios_to_run_all = match_seed_story_scenario_with_entities(SCENARIO_NAMES_TO_STORY_2,
                                                                                 entities_brainstormed_res)
    else:
        story_and_scenarios_to_run_all += match_seed_story_scenario_with_entities(SCENARIO_NAMES_TO_STORY,
                                                                                 entities_brainstormed_res)
        story_and_scenarios_to_run_all += match_seed_story_scenario_with_entities(SCENARIO_NAMES_TO_STORY_2,
                                                                                  entities_brainstormed_res)

    if running_mode == "test":
        story_and_scenarios_to_run_all = story_and_scenarios_to_run_all[:3]
    story_and_scenarios_file = entities_file.replace("1_entity_brainstorm", "2_story_and_scenarios_to_run")
    save_jsonl(story_and_scenarios_file, story_and_scenarios_to_run_all)
    print(f"SAVED story and scenarios to run to {story_and_scenarios_file}")


    ## Step 2: Generate story components
    raw_stories_generated = generate_story_components_raw(INSTRUCTIONS, args.story_models, story_and_scenarios_to_run_all, verbose=False)
    all_stories_raw_filename = os.path.join(output_dir,
                                            f"{file_prefix}_part3_all_stories_raw_entity-models_{'_'.join(args.entity_models)}_story-models_{'_'.join(args.story_models)}.jsonl")
    save_jsonl(all_stories_raw_filename, raw_stories_generated)
    print(f"SAVED {len(raw_stories_generated)} raw model generated stories to {all_stories_raw_filename}")


    ## Step 3: Process generated stories
    stories_processed, _ = process_generated_stories(raw_stories_generated, exclude_seed=args.exclude_seed_stories)
    all_stories_filename = os.path.join(output_dir,
                                             f"{file_prefix}_part4_all_stories_processed_entity-models_{'_'.join(args.entity_models)}_story-models_{'_'.join(args.story_models)}.jsonl")
    save_jsonl(all_stories_filename, stories_processed)
    print(f"SAVED {len(stories_processed)} processed stories to {all_stories_filename}")

if __name__ == '__main__':
    main()