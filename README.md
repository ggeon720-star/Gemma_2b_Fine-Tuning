# QLoRA를 이용한 Gemma-2B의 한양대학교 길안내 특화 파인튜닝 
AI+X 딥러닝 프로젝트

# Members
- 고재윤, 융합전자공학부, jaeyun2448@naver.com
- 권성근, 원자력공학과, gbdlzlemr02@gmail.com
- 신준희, 기계공학부, shinjh0331@naver.com
- 한인권, 기계공학부, humanaeiura1023@gmail.com
  
# Index
1. Proposal
2. Base-model
3. Datasets
4. Methodology
5. Evaluation & Analysis
6. direction for improvement
7. Model use(additional progress)
  
# Proposal
- 동기 및 목표
  
&nbsp; 다들 새내기 때에 가고자 하는 건물까지의 경로를 잘 알지 못해 당황했던 경우가 있었을 것입니다. 저희는 교내 건물과 한양대학교 주변 건물에 대한 정보를 안내해주는 챗봇을 만드는데 목적을 두고 프로젝트를 진행하였습니다. 교내 건물들의 위치를 기반으로 경로 데이터셋을 구성하고 이를 Ko-gemma모델을 base-model로 하여 파인튜닝함으로서 한양대 길안내에 특화된 SLM(Small Language Model)을 구성하는데 초점을 두었습니다.

- 진행 과정 개요

1. 파인튜닝할 Base-model을 선정하고 모델 토크나이저에 맞는 자체적인 한양대 주변 건물 길안내 데이터셋 구축
2. 생성한 데이터셋을 통해 모델 파인튜닝 및 전이학습 진행
3. 모델 학습 결과 분석 및 추론 결과
4. 모델 성능 향상을 위한 개선 방향 제시
5. 모델 활용 프로젝트(additional progress)


# Base-model
&nbsp; 모델 학습을 진행할 수 있는 환경이 Local PC(RTX 3060ti 8GB VRAM)과 Google Colab(T4 GPU 15GB)로 메모리가 한정되어 있기 때문에 큰 LLM모델을 학습하기에는 무리가 있었습니다. 따라서, 학습을 진행하기 위해서 크기가 작으면서도 성능이 준수한 모델을 선정하는 것이 중요하였으며, 이를 결정하기 위해 아래 NVIDIDA에서 제시한 SLM(Small Language Model)모델 별 초당 토큰 수를 비교한 표를 참고하였습니다.

<img width="896" height="484" alt="image" src="https://github.com/user-attachments/assets/e627db24-fff9-4739-8bd6-cfeae036fe64" />

[&nbsp;](https://www.jetson-ai-lab.com/tutorial_slm.html) 

&nbsp; 저희 프로젝트는 한양대학교의 정보, 특히 길안내 정보에 대해서 안내하는 종합 모델을 만드는 것입니다. 따라서, 이 모델을 활용하기 위해서는 인터넷이나 클라우드 시스템이 아닌, Local 임베디드 시스템에 탑재하여 지정된 장소를 기준으로 길을 안내하는 시스템을 구성하려고 하였습니다. 위의 표는 임베디드 시스템인 NVIDIA Jetson orin nano / AGX orin 에서 SLM모델을 작동시키고 측정한 데이터이기에 활용하여 모델을 선택하는 것이 적합하다고 생각하였습니다. 위의 표를 보았을 때, Google의 Gemma모델이 파라미터 개수가 2B이고 초당 토큰 생성 개수가 27/75개로 크기와 성능에 비해 빠르게 작동한다는 것을 알 수 있습니다.

&nbsp; 추가적으로 Gemma-2B모델은 다국어 모델이지만, 한국어 데이터에 대한 학습이 부족하여 한양대학교 길안내라는 특정한 정보에 대한 학습이 부정확할 가능성이 높습니다. 따라서, Gemma-2B-it모델을 한국어 데이터셋으로 파인튜닝한 고려대학교의 Ko-gemma(gemma-ko-2B-v1)모델을 최종 base-model로 선정하여 한양대학교 길안내라는 특정한 한국어 데이터셋에 대한 학습 효율을 높였습니다.

<img width="675" height="279" alt="image" src="https://github.com/user-attachments/assets/2e1480e6-0f2a-494c-b525-90509924d0d5" />

https://github.com/KU-HIAI/Ko-Gemma


# Datasets
&nbsp; 한양대학교 주변 건물들에 대한 길안내 데이터 셋이 존재하지 않기 때문에 자체적으로 제작한 다음 학습을 진행하기로 결정하였습니다. 보다 정확한 데이터셋을 제작하기 위해 Naver API를 기반으로 건물 및 길안내 정보를 수집하고 이를 OPENAI API를 활용해 질문-대답 쌍의 QA데이터셋으로 재구성하여 학습 데이터셋을 생성하였습니다. Naver API의 네이버 지도, 네이버 검색 기능을 활용해 한양대 내부 건물들과 한양대학교 2번출구인 예지문을 기준으로 반경 2km내의 주요 시설들의 이름과 예지문으로부터의 거리, 길안내 경로, 소요 시간 등의 정보를 제공받아 json파일로 1차적인 데이터를 구축하였습니다. 데이터는 다음과 같이 교내 건물 71개, 에지문 기준 반경 1km의 교외 주요 건물 36개, 예지문 기준 반경 1-2km의 교외 주요 건물 62개로 구성하였으며, 건물의 위치 특성에 따라 서로 다른 정보를 담았습니다. 공통적으로는 건물 이름, 소요 시간, 경로 안내 등의 데이터를 담고 있지만, 교내 건물의 경우 건물과 인접한 다른 건물들과의 위치 관계와 건물 번호라는 데이터를, 교외 건물은 건물의 카테고리와 1-2km에 위치한 건물은 대중교통을 이용한 경로 데이터를 추가적으로 구성하였습니다.

<img width="474" height="150" alt="image" src="https://github.com/user-attachments/assets/dfe7f487-46cb-440d-b01d-b03066daf85e" />


&nbsp; Naver API를 이용해 생성한 정보 데이터를 모델 학습을 위한 QA데이터로 재구성하기 위해 OPENAI API를 활용하였습니다. OPENAI API의 GPT-4o-mini의 프롬프트를 통해 한 건물당 40개의 질문-대답 쌍을 생성하였으며, 효율적인 학습을 위해 40개의 질문-대답을 Basic(건물 기본 정보), Route(경로 안내), Location(건물 위치, 주변 건물), Complex(복합 질문) 4개의 type으로 나누어서 구성하였습니다. 이를 통해 정해진 질문이 아닌 다양한 복합 질문에도 자연스러운 응답을 생성할 수 있도록 데이터셋을 구성하였으며, 모델의 overfitting을 방지할 수 있도록 구성하였습니다. 영어 데이터셋은 한국어 데이터셋을 OPENAI API로 번역하여 제작하였으며, Ko-gemma모델의 tokenizer_config.json을 확인해보면, 모델의 학습과 입력 데이터를 message구조로 받는다는 것을 확인하여 이에 맞추어 데이터셋을 messgae구조로 재구성하였으며, 효율적인 학습을 위해 데이터의 순서를 무작위적으로 배치하였습니다. 이 과정을 거쳐 총 13343개의 QA데이터셋을 구성하였으며 모델학습에 사용하였습니다.

<img width="475" height="201" alt="image" src="https://github.com/user-attachments/assets/aca21fb8-d8c4-4026-aa4b-63e2df90dd75" /><img width="309" height="232" alt="image" src="https://github.com/user-attachments/assets/82078196-e280-47b2-bb48-8e6ae61fd689" />




# Methodology 
대략적인 알고리즘
> 1. 패키지 설치
> 2. Google Drive 마운트
> 3. QLoRA 학습 및 LoRA 어댑터 저장
> 4. 학습된 LoRA 어댑터를 Drive에 백업 
> 5. 베이스 모델 및 LoRA 어댑터로 Merged 모델 병합
> 6. 테스트

### 1. 패키지 설치
```python
!pip install -q transformers accelerate bitsandbytes peft trl datasets huggingface_hub ipywidgets
```

### 2. Google Drive 마운트
```python
from google.colab import drive  # Colab에서 Google Drive 연결
drive.mount('/content/drive')
```

### 3. QLoRA 학습 및 LoRA 어댑터 저장
#### 1. 라이브러리 임포트 및 로그인
```python
import os                           # 파일 / 폴더 경로
import json                         # JSON 데이터 읽기 및 쓰기
import random                       # 랜덤 시드 고정
import torch                        # 파이토치 기반 모델

from transformers import (
    AutoTokenizer,                  # 자동으로 모델에 맞는 토크나이저 로드
    AutoModelForCausalLM,           # LLM모델 로드
    BitsAndBytesConfig,             # QLoRA용 4bit / 8bit 양자화 설정
    TrainingArguments,              # 학습 파라미터 설정
    EarlyStoppingCallback,          # 성능 상승 한계 -> 학습 중단
)
from peft import LoraConfig         # 경량 학습 설정 객체
from datasets import Dataset        # 데이터 -> Dataset 객체
from trl import SFTTrainer          # Supervised Fine-Tuning (SFT) 
from huggingface_hub import login   # HuggingFace 모델 다운로드

# Hugging Face 액세스 토큰 
HF_TOKEN = "<YOUR_HF_TOKEN>"
```

#### 2. 라이브러리 임포트 및 로그인
```python
BASE_DIR = "/content/drive/MyDrive/Gemma_2b_Fine-Tuning"            # Google Drive에 저장된 파인튜닝 프로젝트 경로
DATASET_DIR = BASE_DIR

QA_TRAIN_FILES = [
    os.path.join(DATASET_DIR, "train_data_1km_messages.json"),      # 1km 길찾기 Train 데이터
    os.path.join(DATASET_DIR, "train_data_2km_messages.json"),      # 2km 길찾기 Train 데이터
    os.path.join(DATASET_DIR, "train_data_in_messages.json"),       # 학교 내 길찾기 Train 데이터
]

QA_VAL_FILES = [
    os.path.join(DATASET_DIR, "val_data_1km_messages.json"),        # 1km 길찾기 Validation 데이터
    os.path.join(DATASET_DIR, "val_data_2km_messages.json"),        # 1km 길찾기 Validation 데이터
    os.path.join(DATASET_DIR, "val_data_in_messages.json"),         # 1km 길찾기 Validation 데이터
]

MODEL_ID = "nlpai-lab/ko-gemma-2b-v1"                               # HuggingFace에 올라온 한국어로 Fine-tuning된 Gemma 2B 모델
OUTPUT_DIR = "/content/output/gemma-2b-hanyang-guide-final"         # 전체 모델 저장할 경로
ADAPTER_PATH = "/content/output/gemma-2b-hanyang-guide-lora-final"  # LoRA 어댑터 (LoRA 가중치) 저장할 경로

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_PATH, exist_ok=True)
```

#### 3. GPU 확인
```python
if torch.cuda.is_available():
    USE_GPU = True
else:
    USE_GPU = False
```

#### 4. QLoRA 및 LORA 설정
```python
bnb_config = BitsAndBytesConfig(                               
    load_in_4bit=True,                                        # 4bit 정밀도로 모델 로드 
    bnb_4bit_quant_type="nf4",                                # NormalFloat4 양자화 사용 (기존 4bit 방식보다 재현율 높음 -> 품질 더 잘 유지)
    bnb_4bit_compute_dtype=torch.float16,                     # 실제 연산은 FP16으로 수행
    bnb_4bit_use_double_quant=True,                           # Double quantization으로, 양자화된 weight를 1번 더 압축 -> 성능 감소 없이 더 작게 저장
)

lora_config = LoraConfig(
    r=16,                                                     # LoRA Rank 
    lora_alpha=32,                                            # LoRA Scaling Factor (통상적으로 Rank의 2배)
    lora_dropout=0.05,                                        # 과적합 방지
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # LoRA를 적용할 모듈 지정 (Attention 내 Projection만 적용)
    bias="none",                                              # 편향 학습 X
    task_type="CAUSAL_LM",                                    # Casual Language Model로 학습
)
```

#### 5. 모델 및 토크나이저 로드
```python
tokenizer = AutoTokenizer.from_pretrained(                    # HuggingFace에서 MODEL_ID에 해당되는 토크나이저 파일 다운
    MODEL_ID,
    local_files_only=False,
)
tokenizer.padding_side = "right"                              # 패딩 방향 설정 (입력 시퀀스를 오른쪽으로 패딩)

model = AutoModelForCausalLM.from_pretrained(                 # HuggingFace에서 모델 가중치 다운로드 및 로드
    MODEL_ID,
    quantization_config=bnb_config,                           # NF4 기반 QLoRA 설정
    device_map="auto",                                        # GPU 및 CPU 자동감지
    torch_dtype=torch.float16,                                # 계산 시 데이터 타입을 FP16으로 설정
    local_files_only=False,                                   # 로컬에 모델 없을 시 HuggingFace에서 다운 
)
```

#### 6. 데이터셋 로드 (message 포맷)
```python
def load_messages_data(file_paths, dataset_type="Train"):                          # JSON 파일들로 내부 메시지를 텍스트로 변환
    all_texts = []                                                                 # 변환된 텍스트를 누적시킬 리스트
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:                      # UTF-8로 JSON 파일 읽고, 리스트로 변환
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "messages" in item and isinstance(item["messages"], list):
                        try:
                            text = tokenizer.apply_chat_template(                  # 대화형 텍스트 형태로 변환 (message 리스트 -> 하나의 문자열로 합쳐서 사용)
                                item["messages"],
                                tokenize=False,
                                add_generation_prompt=False,
                            )
                            all_texts.append(text)
                        except Exception as e:
                            pass
        except Exception as e:
            pass
    return all_texts

train_texts = load_messages_data(QA_TRAIN_FILES, "Train")                          # Train용 JSON 3개 파일을 읽은 뒤, 텍스트 리스트 생성
val_texts = load_messages_data(QA_VAL_FILES, "Validation")                         # Validation용 JSON 3개 파일을 읽은 뒤, 텍스트 리스트 생성

if not val_texts and train_texts:                                                  # Validation JSON이 없는 경우, Train data를 90/10으로 Split 시키기
    split_idx = int(len(train_texts) * 0.9)
    val_texts = train_texts[split_idx:]
    train_texts = train_texts[:split_idx]

train_dataset = Dataset.from_dict({"text": train_texts}) if train_texts else Dataset.from_dict({"text": []})   # SFTTrainer용 형식으로 변환
eval_dataset = Dataset.from_dict({"text": val_texts}) if val_texts else Dataset.from_dict({"text": []})
```

#### 7. formatting_func 정의
```python
def formatting_func(example):   # 데이터셋의 1개의 샘플을 받아, 모델에 넘길 형태로 가공
    return example["text"]
```



#### 8. SFTTrainer 설정
```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,              # 학습결과 저장할 폴더
    num_train_epochs=3,                 # Epoch
    per_device_train_batch_size=2,      # GPU 1개당 2개 샘플씩 학습에 넣음 (Train)
    per_device_eval_batch_size=2,       # GPU 1개당 2개 샘플씩 학습에 넣음 (Validation)
    gradient_accumulation_steps=8,      # 8번 누적한 뒤 업데이트
    gradient_checkpointing=True,        # 중간 계산을 저장하지 않고 역전파 때 다시 계산 -> 메모리 절약
    max_grad_norm=1.0,                  # Gradient Clipping 방지
    optim="paged_adamw_8bit",           # 옵티마이저 (파라미터 업데이트 규칙) 설정 / paged_adamw_8bit
    learning_rate=2e-4,                 # 학습률
    lr_scheduler_type="cosine",         # Scheduler (학습률 줄이는 함수)를 Cosine로 설정
    warmup_ratio=0.03,                  # 전체 스텝의 3% 구간까지는 학습률 0으로 설정
    weight_decay=0.01,                  # 과적합 방지
    eval_strategy="steps",              
    eval_steps=100,                     # 100 Step마다 Validation 실행
    save_steps=100,                     # 100 Step마다 체크포인트 저장
    save_total_limit=3,                 # 가장 최근의 체크포인트 3개를 제외한 나머지 삭제
    fp16=True,                          # 학습을 Float16으로 수행
    bf16=False,                         
    load_best_model_at_end=True,        # 학습 종료 후 Validation에서 가장 좋은 성능 체크포인트 로드
    metric_for_best_model="eval_loss",  # Eval_Loss로 모델 평가 기준 설정
    greater_is_better=False,
    logging_dir=f"{OUTPUT_DIR}/logs",   # 로그 저장 폴더
    logging_steps=10,                   # 10 Step마다 학습 로그 출력
    report_to="tensorboard",            # Tensorboard에 로그 기록
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=3)              # Eval_Loss가 3번 이상 향상하지 않는다면 학습 중단

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    callbacks=[early_stopping],
)
```




#### 9. Training
```python
trainer.train()                                  # 위의 설정 따라 학습
try:
    trainer.model.save_pretrained(ADAPTER_PATH)  # LoRA 가중치 저장
    tokenizer.save_pretrained(ADAPTER_PATH)      # 토크나이저 저장 (토큰화된 문장)
```


### 4. 학습된 LoRA 어댑터를 Drive에 백업
```python
!mkdir -p /content/drive/MyDrive/Gemma_2B_Trained                                                   # Google Drive에 Gemma_2B_Trained 폴더 생성
!cp -r /content/output/gemma-2b-hanyang-guide-lora-final /content/drive/MyDrive/Gemma_2B_Trained/   # LoRA Adapter 및 토크나이저 저장
```


### 5. 베이스 모델 및 LoRA 어댑터 Merged 모델 병합
```python
os.makedirs(OUTPUT_DIR, exist_ok=True)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                     # Epoch
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.2,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    warmup_steps=5,
    logging_strategy="steps",
    learning_rate=2e-4,                     # 학습률
    fp16=True,
    report_to="tensorboard",
    save_strategy="epoch",
    load_best_model_at_end=False,
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    formatting_func=lambda x: x["text"],
)
```

### 6. 학습 실행 및 LoRA 어댑터 저장, 베이스 모델과 병합
#### 1. 경로 설정
```python
BASE_MODEL = "nlpai-lab/ko-gemma-2b-v1"

# ⬇⬇⬇ 여기 두 줄만 네 드라이브 구조에 맞게 수정해줘 ⬇⬇⬇
ADAPTER_PATH = "/content/drive/MyDrive/Gemma_2b_Fine-Tuning/gemma-2b-hanyang-guide-lora-final"
MERGED_PATH  = "/content/drive/MyDrive/Gemma_2b_Fine-Tuning/gemma-2b-hanyang-final-merged"
# ⬆⬆⬆ 폴더 이름/경로만 정확히 맞추면 됨 ⬆⬆⬆

# 어댑터 경로 확인
if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(f"❌ 어댑터 폴더를 찾을 수 없습니다: {ADAPTER_PATH}")

os.makedirs(MERGED_PATH, exist_ok=True)

print("=" * 70)
print("🔄 Gemma LoRA → Merged 모델 병합 (Colab/GPU 버전)")
print("=" * 70)
print(f"📦 베이스 모델: {BASE_MODEL}")
print(f"🔗 LoRA 어댑터: {ADAPTER_PATH}")
print(f"💾 병합 모델 저장 경로: {MERGED_PATH}")
print("=" * 70 + "\n")
```

#### 2. 디바이스 및 메모리 정보
```python
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠️ GPU를 찾을 수 없습니다. CPU로 병합 시 메모리 부족(OOM)이 날 수 있습니다.")

print()
```

#### 3. 베이스 모델 및 토크나이저 로드
```python
print("1단계: 베이스 모델 로드 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",          # GPU 자동 사용
    torch_dtype=torch.float16,  # fp16으로 메모리 절약
)

print("✅ 베이스 모델 로드 완료\n")

print("2단계: 토크나이저 로드 중...")
# 어댑터 쪽에 저장된 tokenizer를 우선 사용
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(tokenizer))
    print("   ⚠️ pad_token이 없어 새로 추가했습니다.")

print("✅ 토크나이저 로드 완료\n")
```

#### 4. LoRA 어댑터 로드 및 병합
```python
print("3단계: LoRA 어댑터 로드 중...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    device_map="auto"
)
print("✅ LoRA 어댑터 로드 완료\n")

print("4단계: merge_and_unload() 실행 중...")
merged_model = model.merge_and_unload()   # LoRA 가중치를 베이스에 굽기
merged_model.to(device)
print("✅ merge_and_unload() 성공\n")

# (선택) PEFT 관련 속성 정리 - 꼭 없어도 되지만 깔끔하게 정리
attrs_to_remove = [
    'peft_config',
    'active_adapter',
    'active_adapters',
    '_hf_peft_config_loaded',
    'peft_type',
    'base_model_prefix'
]
for attr in attrs_to_remove:
    if hasattr(merged_model, attr):
        try:
            delattr(merged_model, attr)
            print(f"   ✓ {attr} 제거됨")
        except Exception:
            pass

print("✅ 속성 정리 완료\n")
```

#### 5. 병합 모델 저장
```python
print("5단계: 병합된 모델 저장 중...")

try:
    merged_model.save_pretrained(
        MERGED_PATH,
        safe_serialization=True,   # safetensors로 저장
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(MERGED_PATH)
    print(f"✅ 병합 모델이 {MERGED_PATH} 에 저장되었습니다!\n")
except Exception as e:
    print(f"⚠️ safe_serialization 방식 저장 실패: {e}")
    print("   → PyTorch 기본 포맷으로 다시 저장 시도...")
    merged_model.save_pretrained(
        MERGED_PATH,
        safe_serialization=False,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(MERGED_PATH)
    print(f"✅ PyTorch 포맷으로 {MERGED_PATH} 에 저장 완료!\n")

print("=" * 70)
print("✅ 모델 병합 완료 (Colab)")
print("=" * 70)
```

#### 6. 검증
```python
print("\n" + "=" * 70)
print("🧪 저장된 Merged 모델 검증 (간단 테스트)")
print("=" * 70)

try:
    test_tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
    test_model = AutoModelForCausalLM.from_pretrained(
        MERGED_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    test_model.eval()
    print("✅ 저장된 모델 로드 성공!\n")

    from textwrap import shorten

    test_questions = [
        "한양대학교 ERICA 정문에서 제2공학관까지 어떻게 가?",
        "어디에서 학생회관(학생회관 건물)을 찾을 수 있어?"
    ]

    for i, q in enumerate(test_questions, 1):
        print(f"[테스트 {i}] Q: {q}")
        messages = [{"role": "user", "content": q}]
        prompt = test_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = test_tokenizer(prompt, return_tensors="pt").to(test_model.device)

        with torch.no_grad():
            out_ids = test_model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=test_tokenizer.eos_token_id,
            )

        gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
        ans = test_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print("A:", shorten(ans, width=150, placeholder="..."))
        print("-" * 70)

    print("\n✅ 간단 추론 테스트까지 완료!")

except Exception as e:
    print(f"⚠️ 검증 중 오류 발생: {e}")
    print("   (그래도 병합 모델 파일은 MERGED_PATH에 저장되어 있습니다.)")

print("\n최종 저장 경로:", MERGED_PATH)
print("=" * 70)
```

### 7. 테스트
#### 1. 병합 모델 경로 설정
```python
MERGED_MODEL_PATH = "/content/drive/MyDrive/Gemma_2b_Merged"
```

#### 2. 디바이스 설정
```python
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠️ GPU 없음, CPU로 추론합니다. (속도 느릴 수 있음)")
print()
```

#### 3. 토크나이저 & 모델 로드
```python
print("📦 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
print("✅ 토크나이저 로드 완료")
print(f"   BOS: {repr(tokenizer.bos_token)} (ID: {tokenizer.bos_token_id})")
print(f"   EOS: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
print(f"   PAD: {repr(tokenizer.pad_token)} (ID: {tokenizer.pad_token_id})")
print(f"   Chat template 존재 여부: {tokenizer.chat_template is not None}")
print()

print("📦 병합된 모델 로드 중...")
dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,  # GPU 있으면 자동, 없으면 CPU
)
model.eval()

print("✅ 모델 로드 완료")
print(f"   디바이스: {next(model.parameters()).device}")
print("=" * 70 + "\n")
```

#### 4. 응답 생성 함수 (chat template)
```python
def hanyang_guide_chat(
    user_query: str,
    history=None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
):
    """
    병합된 Ko-Gemma 한양 길안내 LLM으로 답변 생성.
    ko-gemma chat_template은 system role을 지원하지 않으므로,
    system_prompt를 첫 user 발화에 텍스트로 포함시키는 방식 사용.
    """
    if history is None:
        history = []

    # 원래 system으로 넣고 싶던 지침을 그냥 텍스트로 포함
    system_prompt = (
        "당신은 한양대학교(서울/ERICA 포함)의 길안내와 건물, 시설 정보를 도와주는 AI입니다. "
        "모르는 정보는 지어내지 말고 '모르겠습니다'라고 답하세요. "
        "길을 설명할 때는 랜드마크를 활용해서 차분하고 구체적으로 설명하세요."
    )

    messages = []

    # 과거 대화 복원 (ko-gemma 템플릿은 user / assistant 조합을 지원)
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    # 이번 질문: system_prompt를 앞에 붙여서 컨텍스트로 줌
    full_user_content = system_prompt + "\n\n" + user_query
    messages.append({"role": "user", "content": full_user_content})

    # Gemma chat template 적용
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 마지막에 <start_of_turn>model\n 추가
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return answer
```

#### 5. 간단 테스트
```python
test_questions = [
    "한양대학교 ERICA 정문에서 제2공학관까지 어떻게 가야 해?",
    "제2공학관 근처에 편의점이나 카페 있어?",
]

print("=" * 70)
print("🧪 간단 테스트")
print("=" * 70)

for i, q in enumerate(test_questions, 1):
    print(f"\n[질문 {i}] {q}")
    ans = hanyang_guide_chat(q)
    print(f"[답변] {ans}")
    print("-" * 70)

print("\n✅ 테스트 완료. 이제 hanyang_guide_chat(질문) 으로 자유롭게 사용할 수 있습상
```

# 5. Evaluation & Analysis
&nbsp; 모델의 성능을 평가하기 위해 훈련시킨 QA데이터 중에서 무작위적으로 3개의 질문을 선택하여 결과를 확인하였습니다. 질문은 간단한 한국어 질문 1개와 복잡한 한국어 질문 1개, 영어 질문 1개를 선택하여 진행하였으며 결과는 다음 사진과 같습니다.

- 원본 데이터에서 질문과 응답
1.
입력 : "한양플라자 건물 번호가 뭐야?"

정답 : "한양플라자의 건물 번호는 105번이야."

2.
입력 : "여기서 비트플렉스약국 어떻게 가?"

정답 : "한양대역 2번 출구로 나와서, 1번 출구 방향으로 길을 건너세요. 그 후 왕십리역 방향으로 약 650m를 직진하면 '왕십리역(비트플렉스)' 건물 2층에 위치해 있어요. 도보로 약 14분 정도 걸려요."

3.
입력 : "What's near the Natural Sciences Building?"

정답 : "The Natural Sciences Building is near the Humanities Hall above and the College of Education below."

- 모델 결과 사진

<img width="928" height="264" alt="image" src="https://github.com/user-attachments/assets/411bbe45-bd81-4d36-9a6e-436be2f73489" />

하지만 모델이 때로는 학습된 정보가 아닌 무작위적인 정보를 제공하는 것을 확인하였습니다. 이를 기반으로 이전의 10개의 질문과 대답을 기억하는 모델 구조를 실험하였습니다.

- 모델 기억 기능 결과 사진

<img width="1224" height="486" alt="image (6)" src="https://github.com/user-attachments/assets/4717a784-7d4a-4407-a1c9-bddfbdd4ab6f" />

결과를 통해 특정한 질문에는 정확한 응답을 생성하는 것을 확인하였으며, 기억 기능도 작동하는 것을 확인하였습니다.

# 6. direction for improvement
&nbsp; 위의 결과와 같이 모델이 학습한 대로 결과를 출력하는 것을 확인할 수 있으며, 학습하지 않은 다양한 질문에도 자연스러운 응답을 생성하는 것을 확인할 수 있었습니다. 하지만 모델이 특정한 소수의 질문에 대해서만 제대로된 응답을 제시하였으며, 대부분의 질문에 대해선 부자연스러운 응답 또는 관련 없는 대답을 제시하는 것을 확인하였습니다. 또한 때때로 특정 응답을 계속 반복하여 제시하기도 하였으며 학습된 데이터와 관련 없는 완전한 오류의 응답을 제시하는 등 성능 면에서 실제로 사용하기에는 한계가 있었습니다. 이러한 문제점을 분석하고 이에 대한 개선방안을 제시하려고 합니다.

### 1. 부정확한 데이터 및 데이터 자체에서 오류가 존재, 주로 짧은 답변 데이터로 구성
   
&nbsp; 데이터셋 자체에서 부정확한 데이터가 존재하며, 영어로 번역된 데이터에서 건물의 이름은 영어로 번역되지 않은 것을 확인하였습니다. 또한, 생성한 QA데이터에서 응답이 20~50토큰으로 매우 짧다는 문제점이 존재하였으며, 조사 결과 11개의 문장은 20토큰 이하로 구성된 것으로 확인하였습니다. 이러한 문제로 인해 부자연스러운 응답이 제시되며 학습에 오류가 생겼을 가능성이 있습니다.

 개선 방안 : 지금의 데이터는 건물 하나당 40개의 QA데이터셋을 구성하였지만 이를 건물 하나당 10~20개의 QA데이터셋으로 축소하고 QA데이터의 응답의 길이를 늘리고 여러 데이터로 구성된 데이터로 변환하여 자연스러운 응답을 제시하고 학습 효율을 높일 수 있습니다. 또한, 경로도 지금은 하나의 경로로만 구성되어 있고 건물의 특성도 한정되어 있지만 이후에 건물의 다양한 특성과 경로를 추가하여 정확한 길안내 모델 데이터셋을 구성하면 성능 향상에 기여할 수 있습니다.

### 2. GPU자원의 한계로 인한 부정확한 학습 및 원본 모델 성능 발휘 불가
   
&nbsp; GPU의 RAM용량이 Local학습에선 RTX 3060Ti(8GB), 코랩에서 T4(15GB)로 LLM파인튜닝에 비해 작은 메모리 용량을 가지고 있기 때문에 제대로된 학습을 진행하지 못했습니다. QLoRA에서 원본 모델을 4비트 양자화하여 불러와 사용하였으며, LoRA Rank도 r=16으로 굉장히 작은 수의 Layer만 선택하여 학습하였기 때문에 제대로된 학습이 불가능하였다고 판단됩니다. 또한, 원본 모델과의 merge를 진행했지만, 파인튜닝 과정에서 4비트로 양자화된 원본 모델을 사용하였기 때문에 merge과정에서 원본 모델의 가중치가 충분히 병합되지 못하고 원본 모델의 기능을 상실했을 가능성이 높습니다. 따라서 원본 모델의 성능을 발휘하지 못하는 망각 현상이 발생하였고 다음 사진과 같이 일상적인 질문에는 정확한 답변을 생성하지 못하는 것을 확인할 수 있습니다.

- 원본 모델 망각 현상
<img width="950" height="290" alt="image (7)" src="https://github.com/user-attachments/assets/0f8e502d-980b-43e0-b0ca-fef582abb3ae" />

 개선 방안 : GPU자원 확보를 통해 QLoRA에서 16비트 또는 QLoRA대신 LoRA만을 사용하여 학습을 진행하면 학습 정확도와 merge과정에서 원본 모델의 기능을 그대로 사용할 수 있습니다. 또한 LoRA Rank를 보다 크게 설정하여 정확한 학습을 진행하여 성능 향상에 기여할 수 있다고 기대됩니다.




# LLM의 성능 평가 기준/방식
## 1. Intrinsic / Extrinsic Evaluation : 모델이 언어를 얼마나 잘 예측하는지를 수치적으로 평가
### - perplextiy
#### <img width="172" height="42" alt="image" src="https://github.com/user-attachments/assets/6155d5ed-3fab-4560-9378-0f369d9841b3" />
#### N은 총 토큰 수 p(x_i)는 x_i번째 정답 토큰을 이 모델이 맞출 확률
#### 평균적으로 모델이 정답 토큰에 대해 얼마나 낮은 혼란도를 갖는지 계산함

## 2. Task-based Evaluation 모델의 실제 문제 해결 능력을 평가하는 지표
### - MMLU
#### 57개 분야의 시험 문제의 정확도를 평가
#### 사람 / GPT-4등의 수준 비교에 사용

### - GSM8K
#### Grade-school math 문제 풀이 정확도를 평가
#### LLM의 수학적 추론을 직접적으로 평가함

### - ARC / HellaSwag / WinoGrande 등...

## 3. Safty / Alignment Evaluation
### - Hallucination Rate
#### 모델의 출력이 사실과 맞지 않을 때의 비율
#### 오류 응답 수 / 전체 응답 수

## 4. Text Genration Quality
### - BLEU
#### n-gram precision 기반
#### <img width="172" height="42" alt="518252181-4e280d64-cc25-4b3b-a31b-2fcd956d9266" src="https://github.com/user-attachments/assets/b5694356-043b-4605-8d55-d306905199de" />
#### p_n은 예측 문장과 참조 문장에서 일치한 수 / 예측문장의 전체 n-gram 수 / w_n은 가중치

## 5. System-level Evaluation

# Related Work (e.g., existing studies)
#### Guo, Z., Jin, R., Liu, C., Huang, Y., Shi, D., Supryadi, Yu, L., Liu, Y., Li, J., Xiong, B., & Xiong, D. (2023, November 25). Evaluating large language models: A comprehensive survey (arXiv pre-print arXiv:2310.19736).
#### 강봉준, & 김영준. (2025). 국내 법률 LLM의 활용과 연구동향 : 환각과 보안 리스크를 중심으로. 산업기술연구논문지, 30(3), 227-240.


# Model use
&nbsp; 제작된 모델은 한양학술타운 프로젝트의 일환으로 사용되었으며, 이에 대해 간단히 소개하려고 합니다.

1. 학술타운 E2E모델 갸요
2. 양자화 
3. 보드 탑재 
4. 결과 

### 학술타운 E2E모델 개요
&nbsp; 제작된 모델은 학술타운 프로젝트로 진행했었던 Whisper모델과 TTS모델을 결합히여 음성 입력에서 음성 출력으로 내보내는 하나의 E2E모델로 구성하는데 사용하였습니다. Whisper는 ASR모델로 음성 입력을 특정 언어로 번역해주는 딥러닝 모델로, 프로젝트에선 한국어 입력을 받기 더 정확히 인식하기 위해서 한국어 데이터셋으로 파인튜닝을 진행하였으며, 작동원리는 다음과 같습니다.

1. Whisper모델과 VAD행

<img width="386" height="209" alt="image" src="https://github.com/user-attachments/assets/9eb82b99-3d48-479e-b26a-05a77039cd18" />

- 양자화 후 용량 비교

<img width="593" height="588" alt="image" src="https://github.com/user-attachments/assets/e15b60e4-d3ab-420a-947c-be7fe5814227" />


- 양자화 후 결과 비교
  
> 원본모델(Gemma-2b-hanyang-final-merged)

<img width="928" height="264" alt="image" src="https://github.com/user-attachments/assets/411bbe45-bd81-4d36-9a6e-436be2f73489" />

> gemma-2b-hanyang-Q4_K_M.gguf

<img width="972" height="235" alt="image" src="https://github.com/user-attachments/assets/49af578d-1bc0-42ae-9891-fabf87ea302e" />

> gemma-2b-hanyang-Q4_0.gguf

<img width="1188" height="256" alt="image" src="https://github.com/user-attachments/assets/8373a25e-ebe5-4c6b-b1c1-3fe72508822e" />

> gemma-2b-hanyang-Q4_K_s.gguf

<img width="777" height="246" alt="image" src="https://github.com/user-attachments/assets/913ef302-6df7-4352-9d82-7bb2c97b83ae" />

### 보드 탑재
&nbsp; NVIDIA Jetson orin nano(8GB) 보드는 1,024개의 CUDA  Core, 32개의 Tensor Core를 가진 AI 추론 및 학습에 특화된 보드로 저전력 및 고속으로 모델을 Local로 동작시키는데 특화되어 있습니다. 모델이 8GB RAM을 가지고 있지만, CPU와 GPU가 하나의 8GB RAM을 공유하기 때문에 RAM용량이 부족하다는 한계가 존재했습니다. 사용가능한 총 RAM 메모리 용량은 6.5GB로 6.5GB내에서 Whisper, VAD, Gemma, TTS모델이 모두 작동할 수 있도록 코드를 설계하였으며, 메모리를 효율적으로 사용하도록 구성하였습니다. 

- 보드 구성
  
<img width="489" height="322" alt="image" src="https://github.com/user-attachments/assets/55fc32dd-f4cb-4702-815e-46aa62606e59" />

- 보드 내 사용 가능한 메모리 용량
  
<img width="995" height="130" alt="image" src="https://github.com/user-attachments/assets/f8fb59fa-d662-403c-a79b-eaa272f7a6b1" />

### 결과
&nbsp; 코드를 구성한 후 모델을 작동시킨 결과는 다음과 같습니다. 모델은 Ubuntu기반 노트북에서 SSH통신을 통해 원격으로 보드 내의 코드를 작동시켰습니다. 사용한 TTS 모델이 영어 base 모델이며 기계음의 부자연스러운 출력이 발생하여 추후에 모델 선정을 통해 교체할 생각입니다.

- 보드 내 모델 로드
    
<img width="686" height="516" alt="image" src="https://github.com/user-attachments/assets/fa20084f-f018-404e-a843-9778da14a375" />

- 실행 결과
  
<img width="621" height="353" alt="image" src="https://github.com/user-attachments/assets/e8732607-5036-4864-bd84-a9e4ea2a697f" />

- 실행 영상
  
https://github.com/user-attachments/assets/cf74bdcc-a74e-4a7b-aecc-20a81747ff87


### 마무리
&nbsp; 이 프로젝트를 통해 상용화가 가능한 E2E모델은 아니었지만, 한양대 음성 길안내 시스템을 구성할 수 있었습니다. 추후에 LLM모델과 TTS모델의 개선을 통해 실제로 설치해서 사용할 수 있는 한양대 Local 음성 챗봇을 만드는 것이 향후 목표입니다.







