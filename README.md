# QLoRA를 이용한 Gemma-2B의 법률 특화 파인튜닝 
AIX 딥러닝 프로젝트

# Members
- 고재윤, (학부), (이메일)
- 권성근, (학부), (이메일)
- 신준희, 기계공학부, shinjh0331@naver.com
- 한인권, (학부), (이메일)

# Index
1. Proposal
2. Datasets
3. Methodology
4. Evaluation & Analysis
5. Related Work
6. Conclusion: Discussion
  
# Proposal
Motivation (Why are you doing this?) :

 해외에서는 LLM 기반 법률 서비스의 상용화가 빠르게 확산되고 있지만, 국내에서는 '데이터 접근성 부족, 개인정보보호법(PIPA)과 같은 규제 장벽, 법조계의 보수적 특성' 등의 이유로 더디게 확산되고 있습니다.
「강봉준 외 1명, 국내 법률 LLM의 활용과 연구동향 : 환각과 보안 리스크를 중심으로」

 특히 국내 법률 AI 도입 과정에서 환각 및 보안 리스크가 단순한 기술적 결함을 넘어 사회적 문제로 연결될 수 있음으로 정확도 이슈를 최소화해야 합니다.
 그렇기에 저희는 기존의 SLM 모델 (Gemma-2B)를 QLoRA를 활용하여 저비용으로 파인튜닝함으로서 더 전문적이고 문맥을 잘 이해하는 LLM을 만들고자 하셨습니다.

What do you want to see at the end? : 

1. 법률 Domain에서의 성능 향상
    - 파인튜닝한 모델의 성능 분석을 위한 평가 기준 필요
    - 기존 Gemma-2B과의 QA 정확도 비교
2. QLoRA (Quantized Low-Rank Adaptation)

# Datasets

# Methodology 
대략적인 알고리즘
> 1. 기본 Import / 환경 변수 설정 / 경로 및 모델 ID 설정
> 2. QLoRA용 설정
> 3. 모델 및 토크나이저 로드
> 4. 법률 JSON 데이터를 'Question', 'Answer', 'Commentary'로 텍스트화 
> 5. SFTTrainer 설정
> 6. 학습 실행 및 LoRA 어댑터 저장

1-1. 기본 Import 및 환경 변수 설정
```python
import torch
import os
import glob
import json
import pandas as pd
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
```
1-2. 경로 및 모델 ID 설정
```python
QA_DATA_DIR = "/content/drive/MyDrive/QA데이터"
MODEL_ID = "RangDev/gemma-2b-it-legal-sum-ko"
BASE_MODEL = MODEL_ID

OUTPUT_DIR = "/content/drive/MyDrive/gemma_law/gemma-2b-law-finetune"
ADAPTER_PATH = "/content/drive/MyDrive/gemma_law/gemma-2b-law-lora-adapter"
MERGED_PATH = "/content/drive/MyDrive/gemma_law/gemma-2b-law-finetuned-merged"
```

2-1. BitsAndBytesConfig (4bit 양자화)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)
```
2-2. LoRA 설정
```python
lora_config = LoraConfig(
    r=16,                                                     # LoRA 랭크
    lora_alpha=32,                                            # LoRA Scaling Factor
    lora_dropout=0.05,                                        # 드롭아웃 비율
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
```

3. 모델 & 토크나이저 로드
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'right'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.resize_token_embeddings(len(tokenizer))
```

4. 법률 JSON 데이터를 'Question', 'Answer', 'Commentary'로 텍스트화
4-1. load_and_format_data 함수 정의
```python
def load_and_format_data(data_dir):
    processed_data = []

    qa_files_pattern = os.path.join(data_dir, "**", "*.json")
    qa_files = glob.glob(qa_files_pattern, recursive=True)
    print(f"총 {len(qa_files)}개의 JSON 파일을 찾았습니다 (하위 디렉토리 포함)...")

    if not qa_files:
        raise ValueError(f"데이터를 찾을 수 없습니다. {data_dir} 경로를 확인하세요.")
    for file_path in qa_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                item = json.load(f)

            question = item.get('question')
            simple_answer = item.get('answer')
            commentary = item.get('commentary') # 없으면 None
            if question and simple_answer:

                full_answer = simple_answer
                if commentary and commentary.strip():
                    full_answer += f"\n\n[근거]\n{commentary}"
                text = f"""<bos><start_of_turn>user
{question}
<end_of_turn>
<start_of_turn>model
{full_answer}<end_of_turn><eos>"""
                processed_data.append({"text": text})
            else:
                 print(f"[경고] {file_path} 파일에 'question' 또는 'answer' 키가 없어 건너뜁니다.")
        except Exception as e:
            print(f"QA 파일 처리 오류 ({file_path}): {e}")
    if not processed_data:
        raise ValueError("학습할 유효한 데이터가 없습니다. JSON 파일 내용을 확인하세요.")

    print(f"총 {len(processed_data)}개의 유효한 학습 데이터를 로드했습니다.")

    df = pd.DataFrame(processed_data)
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    return train_dataset, eval_dataset
```
4-2. 실제 데이터 로드
```python
train_dataset, eval_dataset = load_and_format_data(QA_DATA_DIR)

print(f"\nTrain 셋: {len(train_dataset)}개, Eval 셋: {len(eval_dataset)}개")
print("\n--- 데이터 로드 및 포맷팅 완료 ---")
if len(train_dataset) > 0:
    print("샘플 데이터 (Train):")
    print(train_dataset[0]['text'])
```


# Evaluation & Analysis
- Graphs, tables, any statistics (if any)

# Related Work (e.g., existing studies)
- Tools, libaries, blogs, or any documentatiton that you have used to do this project.

- Conclusion : Discussion
