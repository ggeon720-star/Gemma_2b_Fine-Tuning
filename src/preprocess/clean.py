"""
기존 데이터셋을 Ko-Gemma Chat Template 포맷으로 변환
tokenizer_config.json 기반 정확한 변환
"""

import json
import os

# ========================================================================
# 기존 데이터셋 경로
# ========================================================================

DATASET_DIR = "./dataset_final"

input_files = [
    "train_data_1km.json",
    "train_data_2km.json",
    "train_data_in.json",
    "val_data_1km.json",
    "val_data_2km.json",
    "val_data_in.json"
]

# ========================================================================
# 변환 함수
# ========================================================================

def convert_to_messages_format(old_data):
    """
    기존 포맷을 messages 포맷으로 변환
    
    입력: {"text": "<bos><start_of_turn>user\n질문<end_of_turn>\n<start_of_turn>model\n답변<end_of_turn><eos>"}
    출력: {"messages": [{"role": "user", "content": "질문"}, {"role": "assistant", "content": "답변"}]}
    """
    converted_data = []
    
    for idx, item in enumerate(old_data):
        try:
            text = item['text']
            
            # 모든 특수 토큰 제거
            text = text.replace('<bos>', '').replace('<eos>', '').strip()
            
            # <start_of_turn>으로 분할
            parts = text.split('<start_of_turn>')
            
            user_content = None
            assistant_content = None
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # user 부분 추출
                if part.startswith('user'):
                    user_content = part.replace('user', '', 1).strip()
                    # <end_of_turn> 제거
                    user_content = user_content.replace('<end_of_turn>', '').strip()
                
                # model 부분 추출 (assistant로 변환)
                elif part.startswith('model'):
                    assistant_content = part.replace('model', '', 1).strip()
                    # <end_of_turn> 제거
                    assistant_content = assistant_content.replace('<end_of_turn>', '').strip()
            
            # 유효성 검사
            if user_content and assistant_content:
                # ⭐ messages 포맷으로 변환
                converted_item = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }
                converted_data.append(converted_item)
            else:
                print(f"⚠️  인덱스 {idx}: 유효하지 않은 데이터 (user 또는 assistant 누락)")
        
        except Exception as e:
            print(f"❌ 인덱스 {idx} 변환 실패: {e}")
            continue
    
    return converted_data

# ========================================================================
# 데이터 품질 개선 함수 (옵션)
# ========================================================================

def improve_answer_quality(messages):
    """
    답변이 너무 짧으면 경고 (최소 20자 권장)
    """
    assistant_content = messages[1]["content"]
    
    if len(assistant_content) < 20:
        return f"⚠️  짧은 답변 ({len(assistant_content)}자): {assistant_content[:30]}..."
    
    return None

# ========================================================================
# 변환 실행
# ========================================================================

print("="*80)
print("🔄 데이터셋 변환: 기존 포맷 → messages 포맷")
print("="*80)

for filename in input_files:
    input_path = os.path.join(DATASET_DIR, filename)
    
    if not os.path.exists(input_path):
        print(f"⚠️  파일 없음: {filename}")
        continue
    
    print(f"\n📂 처리 중: {filename}")
    print("-"*80)
    
    # 기존 데이터 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    print(f"원본 데이터: {len(old_data)}개")
    
    # 변환
    converted_data = convert_to_messages_format(old_data)
    print(f"변환 완료: {len(converted_data)}개")
    
    # 품질 체크 (처음 5개만)
    print(f"\n📊 품질 체크 (샘플 5개):")
    warnings = []
    for i, item in enumerate(converted_data[:5]):
        warning = improve_answer_quality(item["messages"])
        if warning:
            warnings.append(f"  #{i}: {warning}")
    
    if warnings:
        print("\n".join(warnings))
    else:
        print("  ✅ 샘플 데이터 품질 양호")
    
    # 저장
    output_filename = filename.replace('.json', '_messages.json')
    output_path = os.path.join(DATASET_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 저장 완료: {output_filename}")
    
    # 샘플 출력
    if len(converted_data) > 0:
        print(f"\n📝 변환 샘플:")
        sample = converted_data[0]["messages"]
        print(f"  Q: {sample[0]['content'][:60]}...")
        print(f"  A: {sample[1]['content'][:60]}...")

print("\n" + "="*80)
print("✅ 모든 파일 변환 완료!")
print("="*80)

print("\n📁 생성된 파일:")
for filename in input_files:
    output_filename = filename.replace('.json', '_messages.json')
    print(f"  - {output_filename}")

print("\n⚠️  다음 단계:")
print("  1. 변환된 파일 확인")
print("  2. 짧은 답변들을 더 상세하게 수정 (권장)")
print("  3. 학습 코드에서 새 파일 사용")

# ========================================================================
# 변환 예시 출력
# ========================================================================

print("\n" + "="*80)
print("📋 변환 예시")
print("="*80)

print("\n❌ 기존 포맷:")
print("""{
  "text": "<bos><start_of_turn>user\\nWhich building is further away, HIT or the FTC?<end_of_turn>\\n<start_of_turn>model\\nHIT is closer to FTC.<end_of_turn><eos>"
}""")

print("\n✅ 새 포맷:")
print("""{
  "messages": [
    {
      "role": "user",
      "content": "Which building is further away, HIT or the FTC?"
    },
    {
      "role": "assistant",
      "content": "HIT is closer to FTC. Haengwon Park is a bit further on the right from HIT."
    }
  ]
}""")
