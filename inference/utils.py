import re
import json
import math

OPENAI_REASONING_SUMMARY_MODELS_OPTIONS = \
    {"gpt-5-2025-08-07": {'temperature': 1.0} # let model reason for as long as it wants, gpt-5 requires temp 1.0
     }

try:
    from openai import OpenAI

    OPENAI_CLIENT = OpenAI()
except Exception:
    OPENAI_CLIENT = None

try:
    import anthropic

    ANTHROPIC_CLIENT = anthropic.Anthropic()
except Exception:
    print("Anthropic (Claude) client not available!")
    ANTHROPIC_CLIENT = None

try:
    from together import Together

    TOGETHER_CLIENT = Together()
except Exception:
    print("Together client not available!")
    TOGETHER_CLIENT = None

def load_jsonl(file_name):
    with open(file_name, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def load_json(file_name):
    with open(file_name, 'r') as file:
        return json.loads(file.read().strip())

def save_jsonl(file_name, data):
    with open(file_name, 'w') as file:
        for d in data:
            file.write(json.dumps(d))
            file.write("\n")
    return file_name

def save_json(file_name, data):
    with open(file_name, 'w') as file:
        file.write(json.dumps(data))

def chatgpt_complete(messages, model, options=None, client=None):
    client = client or OPENAI_CLIENT
    options = options or {}
    res = client.chat.completions.create(model=model, messages=messages, **options)
    return res

# migrate to use responses for gpt-5 and o-series models to get reasoning summary
def chatgpt_responses(messages, model, reasoning=None, options=None, client=None):
    if reasoning is None:
        reasoning = {"effort": "medium", "summary": "auto"}

    client = client or OPENAI_CLIENT
    options = options or {}
    res = client.responses.create(model=model, input=messages, reasoning=reasoning, **options)
    return res

def together_complete(messages, model, options=None, client=None):
    client = client or TOGETHER_CLIENT
    options = options or {}
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        **options
    )
    return res

def claude_complete(messages, model, options=None, client=None):
    client = client or ANTHROPIC_CLIENT
    if options is None:
        new_options = {}
    else:
        new_options = options.copy()
    new_messages = []
    for message in messages:
        if message['role'] == 'system':
            new_options['system'] = message['content']
        else:
            new_messages.append(message)
    if 'logprobs' in new_options:
        del new_options['logprobs']
    if 'n' in new_options:
        if new_options['n'] == 1:
            del new_options['n']
        else:
            raise ValueError("Option n > 1 not valid for Anthropic models.")
    res = client.messages.create(messages=new_messages, model=model, **new_options)
    return res

def llm_complete(messages, model, options=None):
    # Routes to the correct API client based on model name. Add new model routing here.
    if 'claude' in model:
        return claude_complete(messages, model, options)
    elif 'gpt' in model:
        if model in OPENAI_REASONING_SUMMARY_MODELS_OPTIONS:
            return chatgpt_responses(messages, model, options=OPENAI_REASONING_SUMMARY_MODELS_OPTIONS[model])
        return chatgpt_complete(messages, model, options)
    else:
        return together_complete(messages, model, options)


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

def basic_stats(res, cats=['aware', 'action', 'judge', 'action_combo', 'action_combo_faithful']):
    counts = {}
    credits = {}
    nas = {}
    for res1 in res:
        for cat in cats:
            if res1['id'].endswith(cat):
                counts[cat] = counts.get(cat, 0) + 1
                credits[cat] = credits.get(cat, 0) + res1['acc']
                na = 1 if res1['predicted'] not in ['A', 'B'] else 0
                nas[cat] = nas.get(cat, 0) + na
    stats = {}
    for cat in cats:
        if cat not in counts:
            continue
        stats[cat] = {'len': counts[cat], 'acc': credits[cat]/counts[cat], 'na_frac': nas[cat]/counts[cat]}
    return stats

def process_answer(data, answer_prefix="answer"):
    raw = data['llm_output']
    text = llm_output_text(raw)
    if not isinstance(raw, dict):
        raw = raw.model_dump()
    text = re.sub(f"(?ims){answer_prefix}:?", "", text).strip()
    answer_index = None
    prob = 0
    for idx, choice in enumerate(['(A)', '(B)', '(C)', '(D)']):
        if choice in text:
            if answer_index is None:
                answer_index = idx
                break
    if answer_index is None:
        for idx, choice in enumerate(['(A', '(B', '(C', '(D']):
            if choice in text:
                if answer_index is None:
                    answer_index = idx
                else:
                    answer_index = -1
    if answer_index is None:
        for idx, choice in enumerate(['A', 'B', 'C', 'D']):
            if choice in text:
                if answer_index is None:
                    answer_index = idx
                else:
                    answer_index = -1
    if answer_index is not None and answer_index >= 0:
        answer_pred = ['A', 'B', 'C', 'D'][answer_index]
        if 'choices' in raw and 'logprobs' in raw['choices'][0]:
            logprobs = raw['choices'][0]['logprobs']
            if 'content' in logprobs:
                for token in logprobs['content']:
                    if "An" in token['token']:
                        continue   # Skip "Answer" type tokens
                    if answer_pred in token['token']:
                        prob = math.exp(token['logprob'])
                        break
            else:  # Together response
                for token, logprob in zip(logprobs['tokens'], logprobs['token_logprobs']):
                    if "An" in token:
                        continue   # Skip "Answer" type tokens
                    if answer_pred in token:
                        prob = math.exp(logprob)
                        break

    else:
        answer_pred = text
    accuracy = 1 if answer_pred == data['correct'] else 0
    return {"predicted": answer_pred, "probability": prob, "acc": accuracy}

def make_mcq(question, answer_choices):
    choice_labels = ['A', 'B', 'C', 'D', 'E']
    choice_text = " \n".join(f"({label}) {choice}" for label, choice in zip(choice_labels, answer_choices))
    return f"{question} \n{choice_text}"

def question_prompt(story_data, extra_question=None,
                        add_either_option=False,
                        add_neither_option=False,
                        final_prompt="Respond with just"):
    EITHER_CHOICE = "Both (A) and (B) are likely"
    NEITHER_CHOICE = "Neither (A) nor (B) is likely"
    story_text = story_data['story']
    question = story_data['question']
    answer_choices = story_data['choices']['text']
    correct = story_data['answerKey']
    if add_either_option:
        answer_choices.append(EITHER_CHOICE)
    if add_neither_option:
        answer_choices.append(NEITHER_CHOICE)
    prompt_question_text = ""
    if extra_question is not None:
        p_q = extra_question['question']
        p_choices = extra_question['choices']['text']
        if add_either_option:
            p_choices.append(EITHER_CHOICE)
        if add_neither_option:
            p_choices.append(NEITHER_CHOICE)
        answer = extra_question['answerKey']
        prompt_question_text = f"\nQuestion: {make_mcq(p_q, p_choices)}\nAnswer: ({answer})\n"

    prompt_choice_labels = " or ".join(["(A)", "(B)", "(C)", "(D)"][:len(answer_choices)])
    prompt_choice_labels_quoted = re.sub('(\\(.\\))', '"\\1"', prompt_choice_labels)

    prompt = f"""Given the following story, answer the question by giving the correct answer choice, {prompt_choice_labels}.

Story:
{story_text}
{prompt_question_text}
Question: {make_mcq(question, answer_choices)}

What is the correct answer? {final_prompt} {prompt_choice_labels_quoted}
"""

    return {"prompt": prompt, "correct": correct}