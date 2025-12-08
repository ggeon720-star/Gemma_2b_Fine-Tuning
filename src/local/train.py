import torch
import os
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig
from datasets import Dataset
from trl import SFTTrainer
from huggingface_hub import login

# ========================================================================
# 0. HuggingFace 로그인
# ========================================================================
HF_TOKEN = 
try:
    login(token=HF_TOKEN)
    print("✅ HuggingFace 로그인 성공\n")
except Exception as e:
    print(f"⚠️  로그인 실패: {e}\n")

# ========================================================================
# 1. 경로 및 모델 설정
# ========================================================================

print("="*70)
print("🎓 한양대학교 길안내 AI 학습 FINAL (Messages 포맷 - 수정)")
print("="*70)

BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset_final")

QA_TRAIN_FILES = [
    os.path.join(DATASET_DIR, "train_data_1km_messages.json"),
    os.path.join(DATASET_DIR, "train_data_2km_messages.json"),
    os.path.join(DATASET_DIR, "train_data_in_messages.json")
]

QA_VAL_FILES = [
    os.path.join(DATASET_DIR, "val_data_1km_messages.json"),
    os.path.join(DATASET_DIR, "val_data_2km_messages.json"),
    os.path.join(DATASET_DIR, "val_data_in_messages.json")
]

MODEL_ID = "nlpai-lab/ko-gemma-2b-v1"
OUTPUT_DIR = "./output/gemma-2b-hanyang-guide-final"
ADAPTER_PATH = "./output/gemma-2b-hanyang-guide-lora-final"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_PATH, exist_ok=True)

print(f"📦 베이스 모델: {MODEL_ID}")
print(f"💾 출력 경로: {OUTPUT_DIR}")
print("="*70 + "\n")

# ========================================================================
# 2. GPU 확인
# ========================================================================
print("🖥️  시스템 환경 확인")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  GPU를 찾을 수 없습니다. CPU로 실행됩니다.")
print()

# ========================================================================
# 3. QLoRA 설정
# ========================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

print("="*70)
print("📋 학습 설정")
print("="*70)
print(f"모델 크기: 2B parameters")
print(f"LoRA rank: 16")
print(f"LoRA alpha: 32")
print("="*70 + "\n")

# ========================================================================
# 4. 모델 및 토크나이저 로드
# ========================================================================
print(f"📦 모델 로드 중... ({MODEL_ID})")

try:
    # ⭐ 오프라인 모드 설정 (네트워크 문제 시)
    # 이미 다운로드된 캐시가 있다면 사용
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=False,  # True로 변경하면 완전 오프라인
        resume_download=True,    # 중단된 다운로드 재개
    )
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=False,  # True로 변경하면 완전 오프라인
        resume_download=True,    # 중단된 다운로드 재개
    )
    
    print(f"✅ 모델 로드 완료")
    print(f"📝 Chat template 존재: {tokenizer.chat_template is not None}")
    print(f"🔢 Vocab size: {tokenizer.vocab_size:,}")
    print()

except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit(1)

# ========================================================================
# 5. 데이터셋 로드
# ========================================================================
print("="*70)
print("📂 데이터 로드 (messages 포맷)")
print("="*70)

def load_messages_data(file_paths, dataset_type="Train"):
    """messages 포맷 데이터 로드 후 즉시 chat template 적용"""
    all_texts = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path} 파일을 찾을 수 없습니다.")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if "messages" in item and isinstance(item["messages"], list):
                        # ⭐ 즉시 chat template 적용
                        try:
                            text = tokenizer.apply_chat_template(
                                item["messages"],
                                tokenize=False,
                                add_generation_prompt=False
                            )
                            all_texts.append(text)
                        except Exception as e:
                            print(f"⚠️  Chat template 적용 실패: {e}")
                            print(f"   Messages: {item['messages']}")
                    else:
                        print(f"⚠️  잘못된 포맷: {item}")
                
                print(f"✅ {os.path.basename(file_path)}: {len(data)}개 로드")
            else:
                print(f"⚠️  {file_path}의 형식이 올바르지 않습니다.")
        
        except Exception as e:
            print(f"❌ {file_path} 로드 실패: {e}")
    
    print(f"\n📊 총 {dataset_type} 데이터: {len(all_texts)}개")
    return all_texts

# Train 데이터 로드 (이미 포맷팅된 텍스트)
print("\n[Train 데이터]")
train_texts = load_messages_data(QA_TRAIN_FILES, "Train")

# Validation 데이터 로드 (이미 포맷팅된 텍스트)
print("\n[Validation 데이터]")
val_texts = load_messages_data(QA_VAL_FILES, "Validation")

if not val_texts:
    print("⚠️  Validation 데이터가 없어 Train에서 10% 분리")
    split_idx = int(len(train_texts) * 0.9)
    val_texts = train_texts[split_idx:]
    train_texts = train_texts[:split_idx]

# Dataset 변환 (text 필드 사용)
train_dataset = Dataset.from_dict({"text": train_texts})
eval_dataset = Dataset.from_dict({"text": val_texts})

print("\n" + "="*70)
print("📊 최종 데이터셋")
print("="*70)
print(f"Train: {len(train_dataset):,}개")
print(f"Eval:  {len(eval_dataset):,}개")
print(f"Total: {len(train_dataset) + len(eval_dataset):,}개")
print("="*70 + "\n")

# 샘플 확인
if len(train_dataset) > 0:
    print("📝 샘플 데이터:")
    print("-" * 70)
    sample_text = train_dataset[0]['text']
    print(f"포맷팅된 텍스트 (처음 500자):")
    print(sample_text[:500])
    print("...")
    print("-" * 70 + "\n")

# ========================================================================
# 6. ⭐ formatting_func 불필요 (이미 포맷팅됨)
# ========================================================================
# 데이터가 이미 chat template이 적용된 text이므로 formatting_func 불필요!

# ========================================================================
# 7. SFTTrainer 설정
# ========================================================================
print("⚙️  Trainer 설정 중...\n")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # 에폭 설정
    num_train_epochs=3,
    
    # 배치 크기
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    
    # 메모리 최적화
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    
    # 옵티마이저
    optim="paged_adamw_8bit",
    
    # 학습률
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    
    # 평가 전략
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    
    # 정밀도
    fp16=True,
    
    # 최고 모델 선택
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # 로깅
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    report_to="tensorboard",
)

print("="*70)
print("📋 최종 학습 설정")
print("="*70)
print(f"실질 배치: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
total_steps = len(train_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
print(f"예상 스텝: {total_steps:,}")
print(f"학습률: {training_args.learning_rate}")
print("="*70 + "\n")

# 조기 종료
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

# ⭐ SFTTrainer (formatting_func으로 text 반환)
def formatting_func(example):
    """이미 포맷팅된 text를 그대로 반환"""
    return example["text"]

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,  # ⭐ text를 그대로 반환하는 함수
    callbacks=[early_stopping],
)

# ========================================================================
# 8. 훈련 시작
# ========================================================================
print("="*70)
print("🚀 학습 시작!")
print("="*70)
print("💡 적용된 방식:")
print("   ✅ 데이터 로드 시 즉시 chat template 적용")
print("   ✅ dataset_text_field='text' 사용")
print("   ✅ formatting_func 불필요 (이미 포맷팅됨)")
print("="*70 + "\n")

try:
    trainer.train()
    
    print("\n" + "="*70)
    print("✅ 학습 완료!")
    print("="*70)

except KeyboardInterrupt:
    print("\n⚠️  학습 중단됨.")
except Exception as e:
    print(f"\n❌ 학습 오류: {e}")
    import traceback
    traceback.print_exc()

# ========================================================================
# 9. LoRA 어댑터 저장
# ========================================================================
print("\n" + "="*70)
print("💾 LoRA 어댑터 저장")
print("="*70)

try:
    trainer.model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"✅ 저장 완료: {ADAPTER_PATH}")
    
    print("\n" + "="*70)
    print("🎉 학습 완료!")
    print("="*70)
    print(f"📁 LoRA 어댑터: {ADAPTER_PATH}")
    print("\n⚠️  중요: 추론 시에도 tokenizer.apply_chat_template() 사용!")
    print("="*70)
    
except Exception as e:
    print(f"❌ 저장 실패: {e}")

print("\n✅ 스크립트 종료")
print("="*70)
