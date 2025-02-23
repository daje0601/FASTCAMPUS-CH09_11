from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

from datasets import load_dataset
eval_dataset = load_dataset("json", data_files="/root/workspace/test_dataset.json", split="train")
eval_dataset

from tqdm.auto import tqdm 
result = [] 
for idx in tqdm(range(len(eval_dataset))):
    sql_chat_completion = client.chat.completions.create(
    model="lora_adapter1",
    messages=eval_dataset[idx]["messages"][:2],
    temperature=0.1,
    max_tokens=500,
    )
    result.append((eval_dataset[idx]["messages"][2]["content"], sql_chat_completion.choices[0].message.content))

print("완료되었습니다.")

import json

# 결과를 JSON 파일로 저장
output_file = "/root/workspace/text_to_sql/output_results.json"  # 저장할 파일 이름
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"결과가 {output_file} 파일에 저장되었습니다.")
