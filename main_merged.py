"""
Ko-Gemma 추론 코드 (Merged 모델용)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========================================================================
# 1. 모델 경로 설정
# ========================================================================

# ⭐ Merged 모델은 단일 경로만 필요
MERGED_MODEL_PATH = "./output/gemma-2b-hanyang-final-merged"

print("="*70)
print("🎓 한양대학교 길안내 AI - 추론 (Merged Model)")
print("="*70)

# ========================================================================
# 2. 모델 및 토크나이저 로드
# ========================================================================

print(f"📦 Merged 모델 로드: {MERGED_MODEL_PATH}")
print()

try:
    # ⭐ Merged 모델은 바로 로드 (PeftModel 불필요!)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    
    print(f"✅ 토크나이저 로드 완료")
    print(f"   BOS: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"   EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"   PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"   Chat template: {tokenizer.chat_template is not None}")
    print()
    
    # ⭐ Merged 모델 직접 로드
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model.eval()
    
    print("✅ 모델 로드 완료")
    print(f"   디바이스: {next(model.parameters()).device}")
    print()

except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ========================================================================
# 3. 추론 함수 (Chat Template 사용)
# ========================================================================

def generate_response(
    question, 
    max_new_tokens=512, 
    temperature=0.7, 
    top_p=0.9,
    repetition_penalty=1.1
):
    """
    Ko-Gemma의 chat_template을 사용한 응답 생성
    
    chat_template 규칙:
    - <bos>로 시작
    - user → user, assistant → model로 변환
    - add_generation_prompt=True로 <start_of_turn>model\n 추가
    """
    
    # ⭐ messages 포맷으로 입력 구성 (필수!)
    messages = [
        {"role": "user", "content": question}
    ]
    
    # ⭐ chat_template 적용 (필수!)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 추론 시에는 True!
    )
    
    # 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 디코딩 (입력 제외하고 생성된 부분만)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer.strip()

# ========================================================================
# 4. 테스트
# ========================================================================

print("="*70)
print("🧪 테스트 시작")
print("="*70 + "\n")

test_questions = [
    "How do I get to the College of Human Sciences from Aeji Gate?",
    "한양여대 본관에서 행원스퀘어 어떻게 가?",
    "Which building is further away, HIT or the FTC?",
    "507관은 뭐야?",
    "본관은 박물관 어느 쪽에 있어?",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*70}")
    print(f"Question {i}/{len(test_questions)}")
    print(f"{'='*70}")
    print(f"Q: {question}\n")
    
    # Chat template 적용 확인
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"📝 적용된 프롬프트 (처음 200자):")
    print(repr(prompt[:200]))
    print("-" * 70)
    
    # 답변 생성
    response = generate_response(question)
    
    print(f"\nA: {response}")
    print(f"{'='*70}\n")

# ========================================================================
# 5. 대화형 모드
# ========================================================================

print("\n" + "="*70)
print("💬 대화형 모드")
print("="*70)
print("질문을 입력하세요 (종료: 'quit' 또는 'exit')")
print("="*70 + "\n")

while True:
    try:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("👋 종료합니다.")
            break
        
        if not user_input:
            continue
        
        response = generate_response(user_input)
        print(f"\nAI: {response}\n")
        print("-" * 70 + "\n")
    
    except KeyboardInterrupt:
        print("\n👋 종료합니다.")
        break
    except Exception as e:
        print(f"⚠️  오류 발생: {e}\n")
        import traceback
        traceback.print_exc()

print("\n✅ 프로그램 종료")
print("="*70)