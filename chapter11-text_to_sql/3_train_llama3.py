import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# 제로샷 테스트를 하기 위해 모델을 다운받고, 인퍼런스를 실행합니다. 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "allganize/Llama-3-Alpha-Ko-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = ""
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": 'Task:라틴 아메리카/카리브해 지역의 인구가 783(7.5%)가 될 때, 아시아의 인구는 얼마가 될까요?\nSQL Table:CREATE TABLE table_22767 (\n    "Year" real,\n    "World" real,\n    "Asia" text,\n    "Africa" text,\n    "Europe" text,\n    "Latin America/Caribbean" text,\n    "Northern America" text,\n    "Oceania" text\n)\nQuery:'}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


import warnings
warnings.filterwarnings("ignore")

import trl
import torch
import datasets
import transformers

import pandas as pd
from random import randint
from datasets import Dataset, load_dataset

from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig, AutoPeftModelForCausalLM

import wandb
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline)

from huggingface_hub import login

import os
import json
from openai import OpenAI

print(f"PyTorch version       : {torch.__version__}")
print(f"Transformers version  : {transformers.__version__}")
print(f"TRL version           : {trl.__version__}")
print(f"CUDA available        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version      : {torch.version.cuda}")


login(
  token="Huggingface_token", # 여기에 토큰 추가 
)

# dataset = datasets.load_dataset("Clinton/text-to-sql-v1")
dataset = datasets.load_dataset("daje/kotext-to-sql-v1")

def add_length_column(dataset):
    df = dataset.to_pandas()
    df["total_length"] = 0
    for column_name in ["ko_instruction", "input", "response"]:
        num_words = df[column_name].astype(str).str.split().apply(len)
        df["total_length"] += num_words

    return df

df = add_length_column(dataset["train"])

def filter_by_total_length(df, difficulty, number_of_samples, random_state=8888):  # random_state 추가
    if difficulty == "easy":
        return df[df["total_length"].between(10, 100)].sample(n=number_of_samples, random_state=random_state)  # iloc 대신 sample 사용
    elif difficulty == "moderate":
        return df[df["total_length"].between(101, 300)].sample(n=number_of_samples, random_state=random_state)
    elif difficulty == "difficult":
        return df[df["total_length"].between(301, 1000)].sample(n=number_of_samples, random_state=random_state)

print(max(df["total_length"].to_list()), min(df["total_length"].to_list()))

# 일부 데이터만 샘플링하고 싶은 경우 
# easy = filter_by_total_length(df, "easy", 10000)
# medium = filter_by_total_length(df, "moderate", 10000)
# hard = filter_by_total_length(df, "difficult", 2000)
# dataset = pd.concat([easy, medium, hard])
# dataset = dataset.sample(frac=1, random_state=8888)  # random_state 추가
# dataset = Dataset.from_pandas(dataset)
# easy.shape, medium.shape, hard.shape, dataset.shape

# 전체 데이터로 학습을 할 경우 
dataset = Dataset.from_pandas(df)

# trl docs에 보면 이와 같은 방식으로 SFT Trainer용 데이터를 만들 수 있습니다.
# docs에서는 eos_token을 별도로 추가하라는 안내는 없지만, 저자는 습관적으로 eos_token을 붙혀줍니다.
def get_chat_format(element):
    system_prompt = (
        "You are a helpful programmer assistant that excels at SQL. "
        "Below are sql tables schemas paired with instruction that describes a task. "
        "Using valid SQLite, write a response that appropriately completes the request for the provided tables."
    )
    user_prompt = "### instruction:{ko_instruction} ### Input:{input} ### response:"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format_map(element)},
            {"role": "assistant", "content": element["response"]+tokenizer.eos_token},  # 여기 닫는 괄호 추가
        ]
    }

tokenizer.padding_side = 'right'                      

def apply_chat_format(element):
    """
    1) get_chat_format(element)를 호출해서
       messages를 생성한 뒤, Dataset에 저장할 dict로 반환합니다.
    """
    chat_format = get_chat_format(element)  # get_chat_format은 원본 코드 그대로 사용
    return {
        "messages": chat_format["messages"]
    }

def tokenize_messages(element):
    """
    2) apply_chat_template + tokenizer(...)를 통해
       input_ids와 attention_mask를 만들어 반환합니다.
    """
    # 위 단계에서 "messages"가 이미 Dataset에 들어가있다고 가정
    formatted = tokenizer.apply_chat_template(
        element["messages"],  # messages 리스트
        tokenize=False
    )
    outputs = tokenizer(formatted)
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"]
    }

# train과 test 데이터를 0.9와 0.1로 분할합니다.
dataset = dataset.map(
    apply_chat_format,
    batched=False,
    remove_columns=dataset.features,  # 원하시면 제거
)
dataset = dataset.train_test_split(test_size=0.02)
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")

dataset = dataset.map(
    tokenize_messages,
    batched=False,
    remove_columns=["messages"],  # 이제 messages를 더이상 쓰지 않는다면 제거
)

from trl import DataCollatorForCompletionOnlyLM
response_template = "<|start_header_id|>assistant<|end_header_id|>"
# 만약 실제로 줄바꿈까지 포함되어 있다면 \n까지 넣어줘야 합니다.
# response_template = "<|start_header_id|>assistant<|end_header_id|>\n"

response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(
    response_template_ids,
    tokenizer=tokenizer
)

peft_config = LoraConfig(
        lora_alpha=128,                            
        lora_dropout=0.05,                         # Lora 학습 때 사용할 dropout 확률을 지정합니다. 드롭아웃 확률은 과적합 방지를 위해 학습 중 무작위로 일부 뉴런을 비활성화하는 비율을 지정합니다.
        r=256,                                     # Lora의 저차원 공간의 랭크를 지정합니다. 랭크가 높을수록 모델의 표현력이 증가하지만, 계산 비용도 증가합니다.
        bias="none",                               # Lora 적용 시 바이어스를 사용할지 여부를 설정합니다. "none"으로 설정하면 바이어스를 사용하지 않습니다.
        target_modules=["q_proj", "o_proj",        # Lora를 적용할 모델의 모듈 리스트입니다.
                        "k_proj", "v_proj"
                        "up_proj", "down_proj",
                        "gate_proj",
                        ],
        task_type="CAUSAL_LM",                     # 미세 조정 작업 유형을 CAUSAL_LM으로 지정하여 언어 모델링 작업을 수행합니다.
)


args = TrainingArguments(
    output_dir="llama3-260000_ko", # 모델 저장 및 허브 업로드를 위한 디렉토리 지정 합니다.
    num_train_epochs=1,                   # number of training epochs
    # max_steps=100,                          # 100스텝 동안 훈련 수행합니다.
    per_device_train_batch_size=1,          # 배치 사이즈 설정 합니다.
    gradient_accumulation_steps=2,          # 4스텝마다 역전파 및 가중치 업데이트합니다.
    gradient_checkpointing=False,            # 메모리 절약을 위해 그래디언트 체크포인팅 사용합니다.
    optim="adamw_torch_fused",              # 메모리 효율화할 수 있는 fused AdamW 옵티마이저 사용합니다.
    logging_steps=5000,                       # 10스텝마다 로그 기록합니다.
    save_strategy="steps",                  # 매 에폭마다 체크포인트 저장합니다.
    learning_rate=5e-5,                     # 학습률 2e-4로 설정 (QLoRA 논문 기반)합니다.
    bf16=True,                              # 정밀도 설정으로 학습 속도 향상합니다.
    tf32=True,
    max_grad_norm=0.3,                      # 그래디언트 클리핑 값 0.3으로 설정합니다.
    warmup_ratio=0.03,                      # 워밍업 비율 0.03으로 설정 (QLoRA 논문 기반)합니다.
    lr_scheduler_type="constant",           # 일정한 학습률 스케줄러 사용합니다.
    push_to_hub=False,                       # 훈련된 모델을 Hugging Face Hub에 업로드합니다.
    report_to="wandb",                      # wandb로 매트릭 관찰합니다.
)


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    data_collator=collator,
)

# trainer를 학습합니다.
trainer.train()

