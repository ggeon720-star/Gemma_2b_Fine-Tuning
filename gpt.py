import json, os
import re
from openai import OpenAI
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file. Please check your .env file.")

model_name = "gpt-4o-mini"

client = OpenAI(api_key=api_key)

print(f"✓ API key loaded successfully")
print(f"✓ Using model: {model_name}")
print(f"✓ Generating bilingual QA pairs (Korean + English)\n")


def load_input_json(json_path):
    """지정된 경로의 JSON 파일을 읽어 데이터를 반환합니다."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} buildings from {json_path}")
    return data


def generate_qa_batch(building_info, batch_type, batch_num, language="korean", max_retries=3):
    """
    건물 정보를 바탕으로 특정 유형의 QA 쌍을 배치로 생성합니다.
    
    batch_type: 'basic', 'route', 'location', 'complex' 중 하나
    language: 'korean' 또는 'english'
    """
    building_str = json.dumps(building_info, ensure_ascii=False, indent=2)
    
    # 언어별 설정
    if language == "korean":
        lang_instruction = "한국어로"
        lang_note = "반말 위주, 존댓말 일부 혼용"
        example_questions = {
            'basic': '"역사관 어디 있어?", "101관은 뭐야?", "역사관은 몇 층이야?"',
            'route': '"예지문에서 역사관 어떻게 가?", "역사관까지 걸어서 얼마나 걸려?"',
            'location': '"역사관 근처에 뭐 있어?", "본관은 역사관 어느 쪽이야?"',
            'complex': '"역사관에 뭐 있고 어떻게 가?", "역사관이랑 본관 중 어디가 더 가까워?"'
        }
    else:  # english
        lang_instruction = "영어로"
        lang_note = "자연스러운 구어체 영어 (informal/conversational)"
        example_questions = {
            'basic': '"Where is the History Hall?", "What is building 101?", "How many floors does the History Hall have?"',
            'route': '"How do I get to the History Hall from Aeji Gate?", "How long does it take to walk to the History Hall?"',
            'location': '"What\'s near the History Hall?", "Which direction is the Main Building from the History Hall?"',
            'complex': '"What\'s in the History Hall and how do I get there?", "Which is closer, the History Hall or the Main Building?"'
        }
    
    # 배치 유형에 따른 프롬프트 설정
    prompts = {
        'basic': {
            'count': 10,
            'instruction': f"""
            다음 유형의 질문-답변 쌍을 {lang_instruction} 10개 생성해주세요:
            - 건물 이름/위치 질문 (3개)
            - 건물 특징/층수/시설 질문 (4개)
            - 건물 코드 질문 (1개)
            - 건물 카테고리 질문 (2개)
            
            질문 스타일 예시: {example_questions['basic']}
            """
        },
        'route': {
            'count': 12,
            'instruction': f"""
            다음 유형의 질문-답변 쌍을 {lang_instruction} 12개 생성해주세요:
            - 예지문(지하철역 출구)에서 가는 방법 (5개)
            - 소요 시간 질문 (3개)
            - 근처 건물 기준 경로 (4개)
            
            질문 스타일 예시: {example_questions['route']}
            """
        },
        'location': {
            'count': 10,
            'instruction': f"""
            다음 유형의 질문-답변 쌍을 {lang_instruction} 10개 생성해주세요:
            - 주변 건물 질문 (5개)
            - 상대적 위치/방향 질문 (5개)
            
            질문 스타일 예시: {example_questions['location']}
            """
        },
        'complex': {
            'count': 8,
            'instruction': f"""
            다음 유형의 질문-답변 쌍을 {lang_instruction} 8개 생성해주세요:
            - 복합 정보 질문 (건물 특징 + 위치) (3개)
            - 비교 질문 (2개)
            - 부정 질문 (1개)
            - 건물명 변형 질문 (2개)
            
            질문 스타일 예시: {example_questions['complex']}
            """
        }
    }
    
    batch_config = prompts[batch_type]
    
    if language == "korean":
        prompt = f"""
당신은 한양대학교 캠퍼스 안내 전문가입니다.
아래 건물 정보를 바탕으로 학생들이 실제로 물어볼 법한 자연스러운 질문-답변 쌍을 한국어로 생성해주세요.

[건물 정보]
```json
{building_str}
```

[생성 지침]
{batch_config['instruction']}

[중요 규칙]
1. 질문은 구어체로 자연스럽게 작성 ({lang_note})
2. 질문 표현을 최대한 다양하게:
   - 건물명 변형: "{building_info.get('name', '')}", "{building_info.get('building_code', '')}관", 괄호 안 별칭 활용
   - 질문 형식: "~어디야?", "~알려줘", "~어떻게 가?", "~뭐야?", "~찾으려면?" 등
3. 답변은 친절하고 명확하게, 주어진 정보에만 근거
4. 답변 길이: 50-200자 내외
5. 각 QA는 명확하게 구분되는 내용이어야 함 (중복 최소화)
6. route_description의 "현 위치"는 "예지문(한양대역 2번 출구)"을 의미함
7. nearby_buildings의 방향 정보를 적극 활용할 것

[출력 형식]
반드시 다음 JSON 배열 형식으로만 출력하세요:

```json
[
  {{
    "question": "생성된 질문 1",
    "answer": "생성된 답변 1",
    "type": "{batch_type}",
    "building_code": "{building_info.get('building_code', '')}"
  }},
  {{
    "question": "생성된 질문 2",
    "answer": "생성된 답변 2",
    "type": "{batch_type}",
    "building_code": "{building_info.get('building_code', '')}"
  }}
]
```

정확히 {batch_config['count']}개의 QA 쌍을 생성해주세요.
"""
    else:  # english
        prompt = f"""
You are a Hanyang University campus guide expert.
Based on the building information below, generate natural question-answer pairs in English that students would actually ask.

[Building Information]
```json
{building_str}
```

[Generation Guidelines]
{batch_config['instruction']}

[Important Rules]
1. Questions should be written in natural conversational style ({lang_note})
2. Vary question expressions as much as possible:
   - Building name variations: "{building_info.get('name', '')}", "Building {building_info.get('building_code', '')}", use aliases if in parentheses
   - Question formats: "Where is~?", "Can you tell me~?", "How do I get to~?", "What is~?", "How can I find~?" etc.
3. Answers should be clear and friendly, based only on the given information
4. Answer length: 50-200 characters
5. Each QA should be clearly distinct (minimize duplication)
6. The "current location" in route_description means "Aeji Gate (Exit 2 of Hanyang Univ. Station)"
7. Actively use direction information from nearby_buildings

[Output Format]
Output ONLY in the following JSON array format:

```json
[
  {{
    "question": "Generated question 1",
    "answer": "Generated answer 1",
    "type": "{batch_type}",
    "building_code": "{building_info.get('building_code', '')}"
  }},
  {{
    "question": "Generated question 2",
    "answer": "Generated answer 2",
    "type": "{batch_type}",
    "building_code": "{building_info.get('building_code', '')}"
  }}
]
```

Generate exactly {batch_config['count']} QA pairs.
"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=3000
            )
            content = response.choices[0].message.content.strip()
            
            if not content:
                print(f"  ⚠️ Empty response for {language} {batch_type} batch {batch_num} (Attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    sleep(2)
                    continue
                return []
            
            # JSON 파싱
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\[.*?\])', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                parsed_list = json.loads(json_str)
                
                if isinstance(parsed_list, list) and len(parsed_list) > 0:
                    print(f"  ✓ Generated {len(parsed_list)} {language} QAs for {batch_type} batch {batch_num}")
                    return parsed_list
                else:
                    raise ValueError("Parsed JSON is empty or invalid")
            else:
                raise ValueError("No JSON array found in response")
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️ Parse error for {language} {batch_type} batch {batch_num}: {e} (Attempt {attempt + 1})")
            if attempt < max_retries - 1:
                sleep(2)
        except Exception as e:
            print(f"  ❌ API error for {language} {batch_type} batch {batch_num}: {e} (Attempt {attempt + 1})")
            if attempt < max_retries - 1:
                sleep(3)
    
    return []


def generate_qa_pairs_for_building(building_info, language="korean"):
    """
    단일 건물에 대해 40개의 QA 쌍을 생성합니다.
    배치별로 나눠서 생성하여 다양성을 확보합니다.
    """
    building_name = building_info.get('name', 'Unknown')
    lang_label = "🇰🇷" if language == "korean" else "🇺🇸"
    print(f"\n🏢 {lang_label} Processing: {building_name} ({language.upper()})")
    
    all_qa_pairs = []
    
    # 배치별 생성 (총 40개 목표)
    batches = [
        ('basic', 1),      # 10개
        ('route', 1),      # 12개
        ('location', 1),   # 10개
        ('complex', 1),    # 8개
    ]
    
    for batch_type, batch_num in batches:
        qa_batch = generate_qa_batch(building_info, batch_type, batch_num, language)
        all_qa_pairs.extend(qa_batch)
        sleep(1)  # API 레이트 리밋 방지
    
    print(f"  📊 Total generated: {len(all_qa_pairs)} {language} QA pairs")
    return all_qa_pairs


def generate_all_qa_pairs(input_data):
    """
    모든 건물에 대해 한국어와 영어 QA 쌍을 생성합니다.
    """
    korean_qa_pairs = []
    english_qa_pairs = []
    korean_counter = 0
    english_counter = 0
    
    for building in tqdm(input_data, desc="Generating bilingual QA dataset"):
        try:
            # 한국어 QA 생성
            korean_pairs = generate_qa_pairs_for_building(building, language="korean")
            
            if korean_pairs:
                for qa in korean_pairs:
                    qa_pair = {
                        "id": f"KO_QA_{korean_counter:05d}",
                        "language": "korean",
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "type": qa.get("type", "unknown"),
                        "building_code": qa.get("building_code", ""),
                        "building_name": building.get("name", ""),
                        "context": building
                    }
                    korean_qa_pairs.append(qa_pair)
                    korean_counter += 1
            else:
                print(f"⚠️ Warning: No Korean QA pairs generated for {building.get('name', 'Unknown')}")
            
            sleep(2)  # 한영 생성 사이 대기
            
            # 영어 QA 생성
            english_pairs = generate_qa_pairs_for_building(building, language="english")
            
            if english_pairs:
                for qa in english_pairs:
                    qa_pair = {
                        "id": f"EN_QA_{english_counter:05d}",
                        "language": "english",
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "type": qa.get("type", "unknown"),
                        "building_code": qa.get("building_code", ""),
                        "building_name": building.get("name", ""),
                        "context": building
                    }
                    english_qa_pairs.append(qa_pair)
                    english_counter += 1
            else:
                print(f"⚠️ Warning: No English QA pairs generated for {building.get('name', 'Unknown')}")
        
        except Exception as e:
            print(f"❌ Error processing {building.get('name', 'Unknown')}: {e}")
    
    return korean_qa_pairs, english_qa_pairs


def save_qa_pairs_to_json(qa_pairs, output_path, language):
    """생성된 QA 쌍 리스트를 JSON 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {language} QA pairs to: {output_path}")


def print_statistics(korean_qa, english_qa):
    """생성된 QA 데이터셋의 통계를 출력합니다."""
    print("\n" + "="*70)
    print("📊 BILINGUAL DATASET STATISTICS")
    print("="*70)
    
    # 한국어 통계
    print("\n🇰🇷 KOREAN QA PAIRS")
    print("-" * 70)
    print(f"Total: {len(korean_qa)}")
    
    ko_type_counts = {}
    for qa in korean_qa:
        qa_type = qa.get('type', 'unknown')
        ko_type_counts[qa_type] = ko_type_counts.get(qa_type, 0) + 1
    
    print("\n[By Type]")
    for qa_type, count in sorted(ko_type_counts.items()):
        print(f"  {qa_type:12s}: {count:4d} ({count/len(korean_qa)*100:.1f}%)")
    
    ko_building_counts = {}
    for qa in korean_qa:
        building = qa.get('building_name', 'Unknown')
        ko_building_counts[building] = ko_building_counts.get(building, 0) + 1
    
    print(f"\n[By Building]")
    if len(ko_building_counts) > 0:
        print(f"  Total buildings: {len(ko_building_counts)}")
        print(f"  Avg QA per building: {len(korean_qa)/len(ko_building_counts):.1f}")
        print(f"  Min: {min(ko_building_counts.values())}, Max: {max(ko_building_counts.values())}")
    
    # 영어 통계
    print("\n" + "-" * 70)
    print("🇺🇸 ENGLISH QA PAIRS")
    print("-" * 70)
    print(f"Total: {len(english_qa)}")
    
    en_type_counts = {}
    for qa in english_qa:
        qa_type = qa.get('type', 'unknown')
        en_type_counts[qa_type] = en_type_counts.get(qa_type, 0) + 1
    
    print("\n[By Type]")
    for qa_type, count in sorted(en_type_counts.items()):
        print(f"  {qa_type:12s}: {count:4d} ({count/len(english_qa)*100:.1f}%)")
    
    en_building_counts = {}
    for qa in english_qa:
        building = qa.get('building_name', 'Unknown')
        en_building_counts[building] = en_building_counts.get(building, 0) + 1
    
    print(f"\n[By Building]")
    if len(en_building_counts) > 0:
        print(f"  Total buildings: {len(en_building_counts)}")
        print(f"  Avg QA per building: {len(english_qa)/len(en_building_counts):.1f}")
        print(f"  Min: {min(en_building_counts.values())}, Max: {max(en_building_counts.values())}")
    
    # 전체 통계
    print("\n" + "-" * 70)
    print("🌍 COMBINED STATISTICS")
    print("-" * 70)
    print(f"Total QA pairs (Korean + English): {len(korean_qa) + len(english_qa)}")
    print(f"Korean: {len(korean_qa)} ({len(korean_qa)/(len(korean_qa)+len(english_qa))*100:.1f}%)")
    print(f"English: {len(english_qa)} ({len(english_qa)/(len(korean_qa)+len(english_qa))*100:.1f}%)")
    
    print("="*70 + "\n")


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    
    # 입력/출력 경로 설정
    input_json_path = r"C:\Users\jaeyu\Desktop\gemma\LLM\campus_buiding_data3.json"
    korean_output_path = r"C:\Users\jaeyu\Desktop\gemma\LLM\\location_qa_pairs_korean.json"
    english_output_path = r"C:\Users\jaeyu\Desktop\gemma\LLM\\location_qa_pairs_english.json"
    combined_output_path = r"C:\Users\jaeyu\Desktop\gemma\LLM\\location_qa_pairs_combined.json"
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(korean_output_path), exist_ok=True)
    
    try:
        # 1. 입력 JSON 파일 로드
        input_data = load_input_json(input_json_path)
        
        # 2. 한영 QA 쌍 생성
        if isinstance(input_data, list):
            korean_qa, english_qa = generate_all_qa_pairs(input_data)
        elif isinstance(input_data, dict):
            print("Input is a single object, wrapping in a list.")
            korean_qa, english_qa = generate_all_qa_pairs([input_data])
        else:
            print("❌ Error: Input JSON format is not a list or object.")
            korean_qa, english_qa = [], []
        
        if not korean_qa and not english_qa:
            print("❌ No QA pairs were generated. Please check your input data and API.")
            exit(1)
        
        # 3. 샘플 출력
        print("\n" + "="*70)
        print("📝 SAMPLE QA PAIRS")
        print("="*70)
        
        if korean_qa:
            print("\n🇰🇷 Korean Sample (First 2):")
            for i, qa in enumerate(korean_qa[:2], 1):
                print(f"\n[KO Sample {i}]")
                print(f"Q: {qa['question']}")
                print(f"A: {qa['answer']}")
                print(f"Type: {qa['type']}, Building: {qa['building_name']}")
        
        if english_qa:
            print("\n🇺🇸 English Sample (First 2):")
            for i, qa in enumerate(english_qa[:2], 1):
                print(f"\n[EN Sample {i}]")
                print(f"Q: {qa['question']}")
                print(f"A: {qa['answer']}")
                print(f"Type: {qa['type']}, Building: {qa['building_name']}")
        
        print("="*70)
        
        # 4. 통계 출력
        print_statistics(korean_qa, english_qa)
        
        # 5. JSON 파일로 저장 (분리)
        save_qa_pairs_to_json(korean_qa, korean_output_path, "Korean")
        save_qa_pairs_to_json(english_qa, english_output_path, "English")
        
        # 6. 통합 파일도 저장 (선택사항)
        combined_qa = korean_qa + english_qa
        save_qa_pairs_to_json(combined_qa, combined_output_path, "Combined")
        
        print("\n" + "="*70)
        print("✅ Successfully generated bilingual QA dataset!")
        print("="*70)
        print(f"🇰🇷 Korean QA pairs: {len(korean_qa)}")
        print(f"   Target: {len(input_data)} buildings × 40 = {len(input_data) * 40}")
        print(f"   Achievement: {len(korean_qa)/(len(input_data)*40)*100:.1f}%")
        
        print(f"\n🇺🇸 English QA pairs: {len(english_qa)}")
        print(f"   Target: {len(input_data)} buildings × 40 = {len(input_data) * 40}")
        print(f"   Achievement: {len(english_qa)/(len(input_data)*40)*100:.1f}%")
        
        print(f"\n🌍 Total QA pairs: {len(korean_qa) + len(english_qa)}")
        print("="*70)

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_json_path}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()