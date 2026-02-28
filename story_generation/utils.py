import re
from story_generation.prompts import SCENARIO_TEMPLATE, INSTRUCTIONS


def llm_output_text(response):
    if not isinstance(response, dict):
        response = response.model_dump()
    if 'content' in response: # Anthropic
        return response['content'][0]['text']
    if 'output' in response: # OpenAI responses
        # res = {"text:": response['output'][1]['content'][0]['text'], "reasoning_summary": response['output'][0]['summary']}
        # print(res)
        return response['output'][1]['content'][0]['text']
    else:  # OpenAI and Together
        return response['choices'][0]['message']['content']

def fill_template(template, fields):
    res = template
    for key, value in fields.items():
        res = res.replace("{"+key+"}", value)
    return res

def make_prompt_prefix(new_scenario,
                           new_entities=None,
                           key_information = None,
                           story_sentence = None,
                           scenario_template = SCENARIO_TEMPLATE):
    partially_filled = {"scenario": new_scenario}
    if new_entities is not None:
        if not isinstance(new_entities, str):
            entities_string = "\n".join(f"{k} = {v}" for k,v in new_entities.items())
        else:
            entities_string = new_entities
        partially_filled['entities'] = entities_string
    if key_information is not None:
        partially_filled['key_information'] = key_information
    if story_sentence is not None:
        partially_filled['story_sentence'] = story_sentence
    prompt_extra = fill_template(scenario_template, partially_filled)
    prompt_extra = re.sub("(?ms)\\{.*", "", prompt_extra).strip()  # Remove from unfilled variable
    return prompt_extra

def parse_entities(string):
    found = re.findall("(.*)=(.*)", string)
    return {k.strip():v.strip() for k,v in found}

def parse_template(template, story):
    key_values = dict(re.findall('(?ms)(?:^|\n\n)([^\n]+):\\s*\n?\\{(.*?)\\}(?:\n\n|$)', template))
    matches = [(x, list(re.finditer("(?ms)^" + re.escape(x) + ":", story))) for x in key_values.keys()]
    res = {}
    for idx, match in enumerate(matches):
        if len(match[1]) != 1:
            raise ValueError(f"Found {len(match[1])} matches for {match[0]}")
    for idx, match in enumerate(matches):
        start_pos = match[1][0].end()
        if idx == len(matches) - 1:
            end_pos = len(story) + 1
        else:
            end_pos = matches[idx + 1][1][0].start() - 1
        res[key_values[match[0]]] = story[start_pos:end_pos].strip()
    return res

def sentences_to_story(sentences):
    story = []
    for sentence in sentences:
        sent = sentence.strip()
        if not re.match(".*[!.,?;]$", sent):
            sent += "."
        story.append(sent)
    return " ".join(story)

def process_output(output, identifier):
    prefix = make_prompt_prefix(output['scenario_text'], new_entities=output['entities'])
    raw_output = output['output']
    generator_model = "NA"
    if not isinstance(raw_output, str):
        raw_output = llm_output_text(raw_output)
        generator_model = output['output'].get("model", "NA")
    if "SCENARIO:" not in raw_output:
        postfix = re.sub('(?ms).*\n', '', prefix)
        if re.match("(?ms).*^"+postfix+"\n", raw_output): # When response repeats the last key, potentially with some verbiage (Claude)
            raw_output = re.sub("(?ms).*^"+postfix, '', raw_output)
        elif raw_output.startswith(postfix):
            prefix = prefix[:-len(postfix)]
        raw_output = prefix + raw_output
    # Hacky workarounds
    raw_output = re.sub("(?m)^CORRECT ANSWER( \\(.*?doesn't know the KEY INFORMATION\\))?:",
                        "CORRECT ANSWER (Person X doesn't know the KEY INFORMATION):", raw_output)
    raw_output = re.sub("(?m)^COUNTERFACTUAL ANSWER( \\(assum.*? knows the KEY INFORMATION\\))?:",
                        "COUNTERFACTUAL ANSWER (assume Person X actually knows the KEY INFORMATION):", raw_output)
    parsed_output = parse_template(SCENARIO_TEMPLATE, raw_output)
    scenario_name = output['scenario']
    story_example_name = output['story_example']
    orig_key_information = parsed_output['key_information'].strip()
    story_sentence = parsed_output['story_sentence']
    orig_question = parsed_output['question']
    orig_correct_answer = parsed_output['correct_answer']
    orig_incorrect_answer = parsed_output['incorrect_answer']
    entities = parse_entities(parsed_output['entities'])
    entities_lc = {k.lower():v for k,v in entities.items()}
    error_flags = []
    res = []
    for severity in [1,2,3]:
        sev_string = f"sev{severity}"
        is_orig = (severity == 2)
        if not is_orig:
            key_info = parsed_output[sev_string].strip()
            incorrect_answer = parsed_output[f"incorrect_answer_{sev_string}"].strip()
        else:
            key_info = orig_key_information
            incorrect_answer = orig_incorrect_answer
        story = sentences_to_story([key_info, story_sentence])
        res_core = {"id": f"{identifier}_{sev_string}",
                    "scenario_name": scenario_name,
                    "story_prompt_example": story_example_name,
                    "entities": entities,
                    "story": story,
                    "severity": severity,
                    "key_information": key_info,
                    "story_sentence": story_sentence,
                    "generator_model": generator_model,
                    "error_flags": error_flags
                   }
        res_belief = res_core.copy()
        person_x = entities_lc['person x']
        aware_question = f'Is {person_x} likely to be aware that "{key_info}"?'
        res_belief.update({
            "id": res_core['id'] + "_aware",
            "question": aware_question,
            "answer_correct": "No",
            "answer_incorrect": "Yes"
        })
        res.append(res_belief)
        res_action = res_core.copy()
        res_action.update({
            "id": res_core['id'] + "_action",
            "question": orig_question,
            "answer_correct": orig_correct_answer,
            "answer_incorrect": incorrect_answer
        })
        res.append(res_action)
        # Strip trailing whitespace and dashes
        for res1 in res:
            for k, v in res1.items():
                if isinstance(v, str):
                    stripped_v = re.sub("(?ms)[\s-]+$", "", v)
                    if stripped_v != v:
                        res1[k] = stripped_v
    return res

def make_prompt(scenario_story_example, new_scenario, new_entities=None,
                    key_information = None,
                    story_sentence = None,
                    instruction=INSTRUCTIONS, scenario_template = SCENARIO_TEMPLATE):
    scenario_example = fill_template(scenario_template, scenario_story_example)
    prompt = fill_template(instruction, {"scenario_example": scenario_example})
    prompt_extra = make_prompt_prefix(new_scenario, new_entities,
                                          key_information, story_sentence, scenario_template)
    res = prompt.strip() + "\n\n" + prompt_extra
    return res