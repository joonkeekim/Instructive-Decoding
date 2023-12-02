# Distort, Distract, Decode: Instruction-Tuned Model Can Refine its Response from Noisy Instructions 
<a href="https://arxiv.org/abs/2311.00233"><img src="https://img.shields.io/badge/Paper-arXiv:2311.00233-Green"></a>

**Taehyeon Kim***, **Joonkee Kim***, **Gihun Lee***, **Se-Young Yun** <br/>
**\***: Equal Contribution

**🎉 Accepted to Instruction Workshop @ NeurIPS 2023 [[Link](https://openreview.net/forum?id=IqJ3CU3flr)]** 

<!-- [Distort, Distract, Decode: Instruction-Tuned Model Can Refine its Response from Noisy Instructions ](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjbndLpuu2CAxWX0GEKHY3dBOUQFnoECBAQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F2311.00233&usg=AOvVaw0ZOF7zPWlPT11XCYhPTvrr&opi=89978449) -->

## Abstract
<p align="center">
  <img src="./figures/overview.png" width="1394"/>
</p>

> While instruction-tuned language models have demonstrated impressive zero-shot generalization, these models often struggle to generate accurate responses when faced with instructions that fall outside their training set. This paper presents ***Instructive Decoding*** (**ID**), a simple yet effective approach that augments the efficacy of instruction-tuned models. Specifically, ID adjusts the logits for next-token prediction in a contrastive manner, utilizing predictions generated from a manipulated version of the original instruction, referred to as a ***noisy instruction***. This noisy instruction aims to elicit responses that could diverge from the intended instruction yet remain plausible. We conduct experiments across a spectrum of such noisy instructions, ranging from those that insert semantic noise via random words to others like 'opposite' that elicit the deviated responses. Our approach achieves considerable performance gains across various instruction-tuned models and tasks without necessitating any additional parameter updates. Notably, utilizing 'opposite' as the noisy instruction in ID, which exhibits the maximum divergence from the original instruction, consistently produces the most significant performance gains across multiple models and tasks.


## Requirements
### Environmental setup
```
conda create -n id python=3.9
conda activate id
```
Install the necessary packages with:
```
pip install -r requirements.txt
```
### Data preparation
```
mkdir -p data/downloads
mkdir -p data

# SuperNatural Instruction
git clone https://github.com/allenai/natural-instructions.git data/downloads
mkdir -p data/supni
mv data/downloads/tasks data/downloads/splits data/supni/
rm -rf data/downloads/natural-instructions

# MMLU
wget -O data/downloads/mmlu_data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/mmlu
tar -xvf data/downloads/mmlu_data.tar -C data/mmlu
rm -rf data/downloads/mmlu_data.tar
```

Then, you will have a directory structure as follows:
```
Instructive-Decoding
├── data
│   ├── supni
│   │   ├── splits
│   │   └── tasks
│   ├── mmlu
│   │   ├── test
│   │   └── ...
├── scripts
│   ├── run_sni.sh
│   ├── run_mmlu.sh
│   └── ...
├── src
│   ├── run_eval.py
│   ├── base_generator.py
│   └── ...
├── requirements.txt
└── ...
```
### Weights
We used the various models in the paper, and you can load those from Huggingface Hub [[`allenai/tk-instruct-11b-def`](https://huggingface.co/allenai/tk-instruct-11b-def), [allenai/tk-instruct-3b-def](https://huggingface.co/allenai/tk-instruct-3b-def), [`WeOpenML/Alpaca-7B-v1`](https://huggingface.co/WeOpenML/Alpaca-7B-v1),and [bigscience/T0_3B](https://huggingface.co/bigscience/T0_3B)], or utilize weight diff from [open-instruct (OpenSNI-7B)](https://github.com/allenai/open-instruct). <br/>
Note that, for `Tk-Large`, we trained our own from [Tk-instruct repository](https://github.com/yizhongw/Tk-Instruct).

## Run Experiments
If you want to make your own `noisy instruction`, change the instructions of `inst_aware_batchify` in `xxx_generator.py`.

To reproduce our results,
```
bash scripts/run_sni.sh
bash scripts/run_mmlu.sh
```
Important arguments:
- `noisy`: 
- `neg_type`: 
- `eps`: 
- `is_decoder`: 
