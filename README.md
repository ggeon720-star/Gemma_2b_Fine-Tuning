------
HY-건물 데이터셋을 이용한 교내 길안내 LLM 만들기

과정 간단히
-> API를 사용해 데이터셋 정보 생성
raw데이터들은 전부 API를 사용해 얻은 기본 건물 정보들
이를 gpt1, 2, 3.py 코드와 clean.py 통해 train_message와 val_message로 QA데이터셋을 생성함
-> 생성한 데이터셋을 train.py로 학습시켜 모델 생성
-> 이후 merge.py를 통해 모델을 기존 모델과 병합
-> main_merged.py를 통해서 실행











# QLoRA를 이용한 Gemma-2B의 법률 특화 파인튜닝 
AIX 딥러닝 프로젝트

# Members
- 고재윤, 융합전자공학부, jaeyun2448@naver.com
- 권성근, 원자력공학과, gbdlzlemr02@gmail.com
- 신준희, 기계공학부, shinjh0331@naver.com
- 한인권, 기계공학부, humanaeiura1023@gmail.com
  
# Index
1. Proposal
2. Datasets
3. Methodology
4. Evaluation & Analysis
5. Related Work
6. Conclusion: Discussion
  
# Proposal
Motivation (Why are you doing this?) :  

&nbsp; 해외에서는 LLM 기반 법률 서비스의 상용화가 빠르게 확산되고 있지만, 국내에서는 '데이터 접근성 부족, 개인정보보호법(PIPA)과 같은 규제 장벽, 법조계의 보수적 특성' 등의 이유로 더디게 확산되고 있습니다.
「강봉준 외 1명, 국내 법률 LLM의 활용과 연구동향 : 환각과 보안 리스크를 중심으로」

&nbsp; 특히 국내 법률 AI 도입 과정에서 환각 및 보안 리스크가 단순한 기술적 결함을 넘어 사회적 문제로 연결될 수 있음으로 정확도 이슈를 최소화해야 합니다.
 그렇기에 저희는 기존의 SLM 모델 (Gemma-2B)를 QLoRA를 활용하여 저비용으로 파인튜닝함으로서 더 전문적이고 문맥을 잘 이해하는 LLM을 만들고자 하셨습니다.

What do you want to see at the end? : 

1. 법률 Domain에서의 성능 향상
    - 파인튜닝한 모델의 성능 분석을 위한 평가 기준 필요
    - 기존 Gemma-2B과의 QA 정확도 비교
2. QLoRA (Quantized Low-Rank Adaptation)

# Datasets

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
from google.colab import drive
drive.mount('/content/drive')
```

### 3. QLoRA 학습 및 LoRA 어댑터 저장
#### 1. 라이브러리 임포트 및 로그인
```python
import os
import json
import random
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig
from datasets import Dataset
from trl import SFTTrainer
from huggingface_hub import login

# Hugging Face 액세스 토큰 (실제 사용 시 환경변수 등으로 관리 권장)
HF_TOKEN = "<YOUR_HF_TOKEN>"

try:
    login(token=HF_TOKEN)
    print("✅ HuggingFace 로그인 성공\n")
except Exception as e:
    print(f"⚠️  로그인 실패: {e}\n")
```

#### 2. 라이브러리 임포트 및 로그인
```python
print("=" * 70)
print("🎓 한양대학교 길안내 AI 학습 (Colab + QLoRA)")
print("=" * 70)

BASE_DIR = "/content/drive/MyDrive/Gemma_2b_Fine-Tuning"
DATASET_DIR = BASE_DIR

QA_TRAIN_FILES = [
    os.path.join(DATASET_DIR, "train_data_1km_messages.json"),
    os.path.join(DATASET_DIR, "train_data_2km_messages.json"),
    os.path.join(DATASET_DIR, "train_data_in_messages.json"),
]

QA_VAL_FILES = [
    os.path.join(DATASET_DIR, "val_data_1km_messages.json"),
    os.path.join(DATASET_DIR, "val_data_2km_messages.json"),
    os.path.join(DATASET_DIR, "val_data_in_messages.json"),
]

MODEL_ID = "nlpai-lab/ko-gemma-2b-v1"
OUTPUT_DIR = "/content/output/gemma-2b-hanyang-guide-final"
ADAPTER_PATH = "/content/output/gemma-2b-hanyang-guide-lora-final"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_PATH, exist_ok=True)

print(f"📦 베이스 모델: {MODEL_ID}")
print(f"💾 출력 경로: {OUTPUT_DIR}")
print(f"📁 데이터 폴더: {DATASET_DIR}")
print("=" * 70 + "\n")
```

#### 3. GPU 확인
```python
print("🖥️  시스템 환경 확인")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    USE_GPU = True
else:
    print("⚠️  GPU를 찾을 수 없습니다. Colab에서 GPU 런타임을 설정하세요.")
    USE_GPU = False
print()
```

#### 4. QLoRA 및 LORA 설정
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

print("=" * 70)
print("📋 학습 설정 (QLoRA + LoRA)")
print("=" * 70)
print("모델 크기: 2B parameters")
print("LoRA rank: 16")
print("LoRA alpha: 32")
print("=" * 70 + "\n")
```

#### 5. 모델 및 토크나이저 로드
```python
print(f"📦 모델 로드 중... ({MODEL_ID})")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=False,
)
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=False,
)

print("✅ 모델 로드 완료")
print(f"📝 Chat template 존재: {tokenizer.chat_template is not None}")
print(f"🔢 Vocab size: {tokenizer.vocab_size:,}")
print()
```

#### 6. 데이터셋 로드 (message 포맷)
```python
print("=" * 70)
print("📂 데이터 로드 (messages 포맷)")
print("=" * 70)

def load_messages_data(file_paths, dataset_type="Train"):
    """messages 형식 json 파일을 로드하고 chat template로 하나의 text로 변환"""
    all_texts = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path} 파일을 찾을 수 없습니다.")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if "messages" in item and isinstance(item["messages"], list):
                        try:
                            text = tokenizer.apply_chat_template(
                                item["messages"],
                                tokenize=False,
                                add_generation_prompt=False,
                            )
                            all_texts.append(text)
                        except Exception as e:
                            print(f"⚠️  Chat template 적용 실패: {e}")
                            print(f"   Messages: {item['messages']}")
                    else:
                        print(f"⚠️  잘못된 포맷: {item}")

                print(f"✅ {os.path.basename(file_path)}: {len(data)}개 로드")
            else:
                print(f"⚠️  {file_path} 형식이 올바르지 않습니다 (list 아님).")

        except Exception as e:
            print(f"❌ {file_path} 로드 실패: {e}")

    print(f"\n📊 총 {dataset_type} 데이터: {len(all_texts)}개")
    return all_texts

print("\n[Train 데이터]")
train_texts = load_messages_data(QA_TRAIN_FILES, "Train")

print("\n[Validation 데이터]")
val_texts = load_messages_data(QA_VAL_FILES, "Validation")

# Validation 데이터가 없으면 Train에서 10%를 분리
if not val_texts and train_texts:
    print("⚠️  Validation 데이터가 없어 Train에서 10%를 분리합니다.")
    split_idx = int(len(train_texts) * 0.9)
    val_texts = train_texts[split_idx:]
    train_texts = train_texts[:split_idx]

train_dataset = Dataset.from_dict({"text": train_texts}) if train_texts else Dataset.from_dict({"text": []})
eval_dataset = Dataset.from_dict({"text": val_texts}) if val_texts else Dataset.from_dict({"text": []})

print("\n" + "=" * 70)
print("📊 최종 데이터셋 크기")
print("=" * 70)
print(f"Train: {len(train_dataset):,}개")
print(f"Eval:  {len(eval_dataset):,}개")
print(f"Total: {len(train_dataset) + len(eval_dataset):,}개")
print("=" * 70 + "\n")

if len(train_dataset) > 0:
    print("📝 샘플 데이터:")
    print("-" * 70)
    sample_text = train_dataset[0]["text"]
    print("포맷팅된 텍스트 (처음 500자):")
    print(sample_text[:500])
    print("...")
    print("-" * 70 + "\n")
else:
    print("⚠️  Train 데이터가 0개입니다. 경로와 json 구조를 확인하세요.\n")
```

#### 7. formatting_func 정의
```python
def formatting_func(example):
    """text 필드를 그대로 사용"""
    return example["text"]
```



#### 8. SFTTrainer 설정
```python
print("⚙️  Trainer 설정 중...\n")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,

    gradient_checkpointing=True,
    max_grad_norm=1.0,

    optim="paged_adamw_8bit",

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,

    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,

    fp16=True,
    bf16=False,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    report_to="tensorboard",
)

print("=" * 70)
print("📋 최종 학습 설정 요약")
print("=" * 70)
effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
print(f"실질 배치 크기: {effective_batch}")
if len(train_dataset) > 0:
    total_steps = len(train_dataset) * training_args.num_train_epochs // effective_batch
else:
    total_steps = 0
print(f"예상 스텝 수: {total_steps:,}")
print(f"학습률: {training_args.learning_rate}")
print("=" * 70 + "\n")

early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

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
print("=" * 70)
print("🚀 학습 시작")
print("=" * 70)
print("💡 구성 요약:")
print("   - messages 포맷 json 6개 사용")
print("   - tokenizer.apply_chat_template()로 text 생성")
print("   - QLoRA (4bit) + LoRA")
print("=" * 70 + "\n")

if len(train_dataset) == 0:
    print("⚠️  Train 데이터가 0개라 학습을 시작하지 않습니다.")
else:
    try:
        trainer.train()

        print("\n" + "=" * 70)
        print("✅ 학습 완료")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n⚠️  학습이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

#%% ==========================
# 9. LoRA 어댑터 저장
#==============================
print("\n" + "=" * 70)
print("💾 LoRA 어댑터 저장")
print("=" * 70)

try:
    trainer.model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"✅ 저장 완료: {ADAPTER_PATH}")

    print("\n" + "=" * 70)
    print("🎉 전체 파이프라인 완료")
    print("=" * 70)
    print(f"📁 LoRA 어댑터 경로: {ADAPTER_PATH}")
    print("\n⚠️  추론 시에도 tokenizer.apply_chat_template()를 사용해야 합니다.")
    print("=" * 70)

except Exception as e:
    print(f"❌ 저장 실패: {e}")

print("\n✅ 스크립트 종료")
print("=" * 70)
```


### 4. 학습된 LoRA 어댑터를 Drive에 백업
```python
!mkdir -p /content/drive/MyDrive/Gemma_2B_Trained
!cp -r /content/output/gemma-2b-hanyang-guide-lora-final /content/drive/MyDrive/Gemma_2B_Trained/
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
```python
trainer.train()
os.makedirs(ADAPTER_PATH, exist_ok=True)
trainer.model.save_pretrained(ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map='auto',
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, device_map='auto', torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

os.makedirs(MERGED_PATH, exist_ok=True)
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
```

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
# Conclusion : Discussion
