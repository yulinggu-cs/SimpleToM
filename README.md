<h1 align="center">SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs </h1>


<p align="center">
<a href="https://arxiv.org/abs/2410.13648"><img src="https://img.shields.io/badge/arXiv-SimpleToM paper-b31b1b.svg" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/allenai/SimpleToM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-datasets/allenai/SimpleToM-blue" alt="Hugging Face"></a>
</p>

<p align="center">
<a href="#-simpletom-introduction">📖 SimpleToM introduction</a> •  
<a href="#-code">💻 Code</a> •
<a href="#-citation">📜 Citation</a>
</p>


## 📖 SimpleToM introduction

Large language models (LLMs) are increasingly tested for a "Theory of Mind" (ToM) — the ability to attribute mental states to oneself and others. Yet most evaluations stop at explicit belief attribution in classical toy stories or stylized tasks, leaving open the questions of whether LLMs can implicitly apply such knowledge to predict human behavior, or to judge an observed behavior, in diverse scenarios. We introduce 🤖 **SimpleToM, a benchmark that advances ToM evaluation along two novel axes** 🧠. 
- First, it probes 🎯 multiple levels of ToM reasoning 🎯, from mental state inference (explicit ToM) to behavior prediction and judgment (applied ToM). 
- Second, it situates these tasks in 🛒🏥 diverse, everyday scenarios 🏫💼 — such as supermarkets, hospitals, schools, and offices — where information asymmetries naturally arise (e.g., hidden defects in grocery store items, incomplete information in provider–patient interactions, or restricted access to locked devices). SimpleToM contains concise stories (e.g., "The can of Pringles has moldy chips in it. Mary picks up the can in the supermarket and walks to the cashier."), each with three questions that test different degrees of ToM reasoning, asking models to predict: (a) mental states ("Is Mary aware of the mold?"), (b) behaviors ("Will Mary pay for the chips or report the mold?"), and (c) judgments ("Mary paid for the chips. Was that reasonable?"). 

🔥 **Experiments reveal a striking gap** 🔥: state-of-the-art models often reliably infer mental state (a), but fail at applying knowledge about the mental state for secondary predictions, with performance dropping sharply for behavior prediction (b) and further for behavior judgment (c). This exposes a **critical fragility** in LLMs’ social reasoning in terms of what they know (explicit ToM) versus how well they can implicitly apply that knowledge for predictions (applied ToM). By uniting assessment of different levels of ToM reasoning with diverse, everyday scenarios, **SimpleToM opens new opportunities for rigorously evaluating and diagnosing ToM abilities in LLMs, and reveals surprising, new insights about current model capabilities, guiding efforts toward future generations of models capable of robust social understanding.**

These are further described in our paper "SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs" (Arxiv Link: https://arxiv.org/abs/2410.13648) **[To appear at ICLR 2026]**


### 📚 Dataset
We make our SimpleToM dataset publicly available at https://huggingface.co/datasets/allenai/SimpleToM.


## 💻 Code

In this repository, we make the code for generating SimpleToM stories and running inference on them publicly available for other researchers to build upon our work.

### Part 1: Creating a conda environment and installing the requirements

We recommend the following steps to setting up an environment for this project:

#### Install conda (if you don't have it)
```
$ wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
$ chmod +x Anaconda*.sh
$ sh Anaconda*.sh
$ export PATH=/home/$USER/anaconda3/bin/:$PATH
```

#### Create a conda environment
The following code snippet creates a new conda environment for this project and installs the requirements inside the conda environment. The requirements.txt file can be found in this repository.

```
$ conda create -n SimpleToM python=3.11
$ conda activate SimpleToM
```

(This is uncommon, but if your conda environment does not come with pip, you may run the following. Otherwise, skip to the next line and it should start installing the requirements.)
```
$ conda install pip
```

```
$ python3.11 -m pip install -r requirements.txt
```

### Part 2: Using your API Keys

#### Set your ‘OPENAI_API_KEY’ Environment Variable using zsh

The following code snippet is for the Linux / MacOS Set-up, and references https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety. You can also find how to do it for other systems as well as alternative ways there.

1. Run the following command in your terminal, replacing yourkey with your API key
```
$ echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
```

2. Update the shell with the new variable
```
$ source ~/.zshrc
```

3. Confirm that you have set your environment variable using the following command
```
$ echo $OPENAI_API_KEY
```
The value of your API key will be the resulting output.

Repeat the same for ANTHROPIC_API_KEY and TOGETHER_API_KEY with your respective keys. 


Now when you open a Jupyter notebook or use python, you’d be able to access your stored key using:
```
os.getenv("OPENAI_API_KEY")
```

You can also check that you can run inference on the various families of models using: 
```commandline
python inference/test_models.py
```
which queries OpenAI models via the OpenAI API, Claude models via the Anthropic API and Llama models via the Together AI API.

### Part 3: Running inference on SimpleToM
Example commands

```commandline
python inference/run_inference.py --models gpt-5-2025-08-07 --subset all --limit 2 --use-cot True
```
If an output file for this command already exists in the output folder, the stored results will be loaded. This behavior is intentional and prevents accidental overwriting of previously saved results. We also store intermediate inference outputs, so that previously ran inference will be loaded and the inference running continues from there. 

To save the output under a different name, provide the optional --output-tag argument with a tag of your choice, e.g.:

```commandline
 python inference/run_inference.py --models gpt-5-2025-08-07 --subset all --limit 2 --use-cot True --output-tag rerun1
```


```commandline
python inference/run_inference.py --models claude-3-haiku-20240307 --subset mental-state-qa --limit 2 --use-cot False --output-tag rerun1
```

```commandline
python inference/run_inference.py --models meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo claude-3-haiku-20240307 --subset mental-state-qa --limit 1 --output-tag rerun1
```

```commandline
python inference/run_inference.py --models gpt-5-2025-08-07 --subset mental-state-qa --output-tag rerun1
```

Note that only the `--models` argument is required. Other arguments are optional, when not set `--subset` defaults to `all`, `--limit` defaults to `None` and `--use-cot` defaults to `False`.

You can find example outputs in sample_outputs.

### Part 4: Using our pipeline to generate new stories 

If you would like to use new models or seed data to generate different stories, from the top level SimpleToM directory, you may use a command like
```commandline
python -m story_generation.run_generation \
  --entity-models gpt-4-0125-preview \
  --story-models gpt-4-0125-preview claude-opus-4-6 \
  --story-seed-round 1 \
  --exclude-seed-stories True \
  --running-mode test \
  --file-prefix test1_runs
```
Run with `--running-mode test` to check that your settings works as intended first. You may change `--entity-models` and `--story-models` arguments to use models of your choice, the ones included in this README are for demonstration purposes only. You may use any `--file-prefix` of your choice too.

To run with all available seed stories, keep seed stories in output, and in full mode (not just sanity testing), run
```commandline
python -m story_generation.run_generation \
  --entity-models gpt-4-0125-preview \
  --story-models gpt-4-0125-preview claude-opus-4-6 \
  --story-seed-round 0 \
  --exclude-seed-stories False
  --file-prefix complete_new_run
```

This generation pipeline allows us to leverage the generative strength of language models to obtain concise stories with
varied entities and diverse situations, suitable for testing different levels of ToM reasoning. The
generated stories (and answer options) were then rigorously filtered by careful human annotators
who passed a strict qualification test, as described in Section 3.2 of our paper. The result is a high-quality and diverse dataset, SimpleToM.


# 📜 Citation

```
@misc{gu2024simpletomexposinggapexplicit,
      title={SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs}, 
      author={Yuling Gu and Oyvind Tafjord and Hyunwoo Kim and Jared Moore and Ronan Le Bras and Peter Clark and Yejin Choi},
      year={2024},
      eprint={2410.13648},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.13648}, 
}
```
(ICLR citation coming soon...)

 
