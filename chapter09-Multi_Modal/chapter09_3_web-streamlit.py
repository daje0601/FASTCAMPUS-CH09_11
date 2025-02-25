import torch 
import pandas as pd
from PIL import Image
import streamlit as st
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForVision2Seq


model_id = "Qwen/Qwen2-VL-7B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(model_id)

# 로라 어댑터가 있는 경로
adapter_path1 = "./qwen2-7b-instruct-harmful-detector-2000/checkpoint-33"

# 첫 번째 Adapter 로드
model.load_adapter(
    adapter_path1,
    adapter_name="adapter1") 
model.set_adapter("adapter1")

def generate_description(messages, model, processor):
   # 추론을 위한 준비
   text = processor.apply_chat_template(
       messages, tokenize=False, add_generation_prompt=True
   )
   image_inputs, video_inputs = process_vision_info(messages)
   inputs = processor(
       text=[text],
       images=image_inputs,
       videos=video_inputs,
       padding=True,
       return_tensors="pt",
   )
   inputs = inputs.to(model.device)
   # 추론: 출력 생성
   generated_ids = model.generate(
      **inputs,            # 앞서 만든 전처리 결과(토큰, 이미지, 비디오 텐서 등)를 전달
      max_new_tokens=128,  # 최대 128토큰 새롭게 생성 
      top_p=0.95,          # 상위 95% 누적 확률에 속하는 후보만 고려하는 Top-p 샘플링
      do_sample=True,      # 확률적 샘플링을 활성화하여 답변을 생성(좀 더 랜덤성을 포함하게 됨)    
      temperature=0.1      # 값이 낮을수록 보수적인(가장 확률 높은 후보에 집중) 출력을 생성하고, 높으면 다양한 출력을 시도
      )
   
   # 모델이 생성한 시퀀스에는 원본 입력 토큰 + 새로 생성된 토큰이 함께 있음 
   # len(in_ids)만큼 잘라내고 나머지 토큰만 가져옴으로써, 새로 생성된 토큰 부분만 분리
   generated_ids_trimmed = [
      out_ids[len(in_ids) :] 
      for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
   # 숫자들을 다시 우리가 인식할 수 있는 글자로 변환 
   output_text = processor.batch_decode(
       generated_ids_trimmed, 
       skip_special_tokens=True,              # 모델 내부적으로 사용하는 특수 토큰을 제거
       clean_up_tokenization_spaces=False     # 토크나이저는 텍스트를 토큰으로 변환할 때, 단어 사이의 공백이나 구두점 처리 등에서 추가적인 공백이 생길 수 있는데, 이를 제거할 것인 말지를 결정함
   )
   return output_text[0]

def main():
    st.title("이미지 & 텍스트 입력을 통한 모델 예측 데모")

    # CSV 파일 불러오기: image_path, text 두 컬럼이 있다고 가정
    df = pd.read_csv('./data/final_df.csv')
    df["is_hate"] = df["is_hate"].astype(int)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    hf_dataset = Dataset.from_pandas(df)
    st.write("## 1. 데이터프레임 내 샘플 선택")
    # 데이터프레임에서 행 선택 (index 기준)
    selected_index = st.selectbox(
        "샘플을 선택하세요",
        df.index
    )
    
    # 선택된 행의 정보 가져오기
    selected_image_path = df.loc[selected_index, 'file_path']
    selected_text = df.loc[selected_index, 'translated']

    st.write("## 2. 이미지와 기본 텍스트 확인하기")
    # 이미지 표시
    try:
        image = Image.open(selected_image_path)
        st.image(image, caption="선택된 이미지", use_column_width=True)
    except Exception as e:
        st.error(f"이미지를 불러오는 중 오류가 발생했습니다: {e}")
    
    st.write("기본 텍스트:", selected_text)

    st.write("## 3. 텍스트 수정 후 예측")
    # 텍스트 입력(기본값은 선택된 텍스트)
    user_input_text = st.text_input("텍스트를 입력하세요", value=selected_text)

    # 예측 버튼
    if st.button("예측 실행"):
        # 실제 모델 로직 수행(여기서는 더미 함수 호출)
        prediction = generate_description(selected_image_path, user_input_text)
        
        # 예측 결과를 0 또는 1에 따라 치환
        if prediction == 1:
            result_text = "혐오 데이터입니다."
        else:
            result_text = "비혐오 데이터입니다."
        
        # 결과 출력
        st.write("### 예측 결과")
        st.write(f"모델 예측 결과: **{result_text}**")


if __name__ == "__main__":
    main()
