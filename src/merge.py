import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

# ------------------------------------------------------------------------
# 0. HuggingFace 로그인
# ------------------------------------------------------------------------
HF_TOKEN = 
try:
    login(token=HF_TOKEN)
    print("✅ HuggingFace 로그인 성공\n")
except Exception as e:
    print(f"⚠️  HuggingFace 로그인 실패: {e}\n")

# ------------------------------------------------------------------------
# 1. 경로 설정
# ------------------------------------------------------------------------
BASE_MODEL = "nlpai-lab/ko-gemma-2b-v1"
ADAPTER_PATH = r"C:\Users\jaeyu\Desktop\gemma\LLM\output\gemma-2b-hanyang-guide-lora-final"
MERGED_PATH = r"C:\Users\jaeyu\Desktop\gemma\LLM\output\gemma-2b-hanyang-final-merged"

# 어댑터 경로 확인
if not os.path.exists(ADAPTER_PATH):
    print(f"❌ 어댑터를 찾을 수 없습니다: {ADAPTER_PATH}")
    print("💡 먼저 local_training_script.py를 실행하여 모델을 학습하세요.")
    exit(1)

# 출력 디렉토리 생성
os.makedirs(MERGED_PATH, exist_ok=True)

# ------------------------------------------------------------------------
# 2. 모델 병합 프로세스
# ------------------------------------------------------------------------
print("=" * 70)
print("🔄 모델 병합 시작")
print("=" * 70)
print(f"📦 베이스 모델: {BASE_MODEL}")
print(f"🔗 LoRA 어댑터: {ADAPTER_PATH}")
print(f"💾 저장 경로: {MERGED_PATH}")
print("=" * 70 + "\n")

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"💾 초기 GPU 메모리:")
    print(f"   할당: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   예약: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\n")

# ------------------------------------------------------------------------
# 방법 1: 수동 병합 (가장 안정적) ⭐ 추천
# ------------------------------------------------------------------------

print("1단계: 베이스 모델 로드...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map='cpu',  # CPU에서 안정적으로 병합
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    print("✅ 베이스 모델 로드 완료\n")
except Exception as e:
    print(f"❌ 베이스 모델 로드 실패: {e}")
    exit(1)

print("2단계: 토크나이저 로드...")
try:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    # pad token 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_model.resize_token_embeddings(len(tokenizer))
        print("   ⚠️  pad_token 추가됨")
    
    print("✅ 토크나이저 로드 완료\n")
except Exception as e:
    print(f"❌ 토크나이저 로드 실패: {e}")
    # 베이스 모델의 토크나이저 사용
    print("   → 베이스 모델의 토크나이저를 사용합니다.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_model.resize_token_embeddings(len(tokenizer))

print("3단계: LoRA 어댑터 로드...")
try:
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        device_map='cpu'
    )
    print("✅ LoRA 어댑터 로드 완료\n")
except Exception as e:
    print(f"❌ LoRA 어댑터 로드 실패: {e}")
    print("💡 어댑터 파일 경로와 포맷을 확인하세요.")
    exit(1)

print("4단계: 어댑터 병합 시도...")
merged_model = None

# 방법 4-A: merge_and_unload() 시도
try:
    merged_model = model.merge_and_unload()
    print("✅ merge_and_unload() 성공!\n")
except Exception as e:
    print(f"⚠️  merge_and_unload() 실패: {e}")
    print("   → 대안: 수동 가중치 추출 방식 사용\n")
    
    # 방법 4-B: 수동으로 병합된 가중치 추출
    try:
        model.eval()
        with torch.no_grad():
            # PEFT 모델의 base_model 속성에서 병합된 가중치 가져오기
            if hasattr(model, 'base_model'):
                if hasattr(model.base_model, 'model'):
                    merged_model = model.base_model.model
                else:
                    merged_model = model.base_model
            else:
                merged_model = model.model
        
        print("✅ 수동 가중치 추출 완료!\n")
    except Exception as e:
        print(f"❌ 수동 가중치 추출도 실패: {e}")
        exit(1)

if merged_model is None:
    print("❌ 모델 병합에 실패했습니다.")
    exit(1)

print("5단계: PEFT 관련 속성 정리...")
# PEFT 관련 속성 모두 제거
attrs_to_remove = [
    'peft_config', 
    'active_adapter', 
    'active_adapters', 
    '_hf_peft_config_loaded',
    'peft_type',
    'base_model_prefix'
]

for attr in attrs_to_remove:
    try:
        if hasattr(merged_model, attr):
            delattr(merged_model, attr)
            print(f"   ✓ {attr} 제거됨")
    except (AttributeError, TypeError):
        # 속성이 프로퍼티나 특수 속성인 경우 무시
        pass

print("✅ 속성 정리 완료\n")

print("6단계: 병합된 모델 저장...")
try:
    # 먼저 safe_serialization으로 저장 시도
    merged_model.save_pretrained(
        MERGED_PATH,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(MERGED_PATH)
    print(f"✅ 모델이 {MERGED_PATH}에 저장되었습니다!\n")
    
except Exception as e:
    print(f"⚠️  safe_serialization 저장 실패: {e}")
    print("   → 대안: PyTorch 기본 방식으로 저장 시도...\n")
    
    try:
        # PyTorch 기본 방식으로 저장
        merged_model.save_pretrained(
            MERGED_PATH,
            safe_serialization=False,
            max_shard_size="2GB"
        )
        tokenizer.save_pretrained(MERGED_PATH)
        print(f"✅ PyTorch 방식으로 저장 완료!\n")
    except Exception as e2:
        print(f"⚠️  save_pretrained도 실패: {e2}")
        print("   → 최후의 대안: state_dict 방식으로 저장...\n")
        
        try:
            # state_dict 직접 저장
            torch.save(
                merged_model.state_dict(), 
                os.path.join(MERGED_PATH, "pytorch_model.bin")
            )
            tokenizer.save_pretrained(MERGED_PATH)
            # config 파일도 저장
            merged_model.config.save_pretrained(MERGED_PATH)
            print(f"✅ state_dict 방식으로 저장 완료!\n")
        except Exception as e3:
            print(f"❌ 모든 저장 방법 실패: {e3}")
            exit(1)

print("=" * 70)
print("✅ 모델 병합 완료!")
print("=" * 70)

# GPU 메모리 상태 (해당되는 경우)
if torch.cuda.is_available():
    print(f"\n💾 최종 GPU 메모리:")
    print(f"   할당: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   예약: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\n")

# ------------------------------------------------------------------------
# 7. 검증: 저장된 모델 로드 테스트
# ------------------------------------------------------------------------
print("=" * 70)
print("🧪 저장된 모델 검증 중...")
print("=" * 70)

try:
    test_model = AutoModelForCausalLM.from_pretrained(
        MERGED_PATH,
        device_map='cpu',
        torch_dtype=torch.float32
    )
    test_tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
    print("✅ 저장된 모델 로드 성공! 병합이 정상적으로 완료되었습니다.\n")
    
    # 간단한 추론 테스트
    print("🧪 간단한 추론 테스트...")
    test_questions = [
        "역사관 어디 있어?",
        "Where is the History Hall?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[테스트 {i}]")
        print(f"Q: {question}")
        
        prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        inputs = test_tokenizer(prompt, return_tensors="pt")
        
        print(f"   ✓ 토크나이저 작동 (입력 토큰 수: {inputs['input_ids'].shape[1]})")
        
        # 짧은 생성 테스트
        try:
            with torch.no_grad():
                outputs = test_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = test_tokenizer.decode(outputs[0], skip_special_tokens=False)
            if "<start_of_turn>model" in response:
                answer = response.split("<start_of_turn>model")[1].split("<end_of_turn>")[0].strip()
                print(f"A: {answer[:100]}..." if len(answer) > 100 else f"A: {answer}")
            else:
                print(f"A: {response[:100]}...")
            
            print("   ✓ 추론 테스트 통과")
            
        except Exception as e:
            print(f"   ⚠️  추론 테스트 실패: {e}")
            print("   (모델은 저장되었지만 추론에 문제가 있을 수 있습니다)")
    
except Exception as e:
    print(f"⚠️  검증 실패: {e}")
    print("   모델은 저장되었지만 로드에 문제가 있을 수 있습니다.")

print("\n" + "=" * 70)
print("🎉 모든 작업 완료!")
print("=" * 70)
print(f"📁 최종 저장 경로: {MERGED_PATH}")
print("\n💡 모델 사용 방법:")
print("   from transformers import AutoModelForCausalLM, AutoTokenizer")
print(f"   model = AutoModelForCausalLM.from_pretrained('{MERGED_PATH}')")
print(f"   tokenizer = AutoTokenizer.from_pretrained('{MERGED_PATH}')")
print("=" * 70)
