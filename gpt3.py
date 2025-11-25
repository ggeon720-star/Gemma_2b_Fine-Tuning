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
print(f"✓ Generating bilingual QA pairs for MID-DISTANCE locations (1-2km, Public Transport)")
print(f"  Focus: Bus/Subway routes, stop IDs, transfer info\n")


def load_input_json(json_path):
    """지정된 경로의 JSON 파일을 읽어 데이터를 반환합니다."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} mid-distance locations from {json_path}")
    return data


def generate_qa_batch(location_info, batch_type, batch_num, language="korean", max_retries=3):
    """
    1-2km 거리 장소의 대중교통 경로 정보를 바탕으로 QA 쌍을 생성합니다.
    
    batch_type: 'basic', 'transit', 'route_detail', 'complex' 중 하나
    language: 'korean' 또는 'english'
    """
    location_str = json.dumps(location_info, ensure_ascii=False, indent=2)
    
    # 언어별 설정
    if language == "korean":
        lang_instruction = "한국어로"
        lang_note = "반말 위주, 존댓말 일부 혼용"
        example_questions = {
            'basic': '"무학중학교 어디 있어?", "무학중학교는 뭐야?", "무학중학교 얼마나 멀어?"',
            'transit': '"무학중학교 버스 타고 가야 해?", "무학중학교 가는 버스 뭐야?", "무학중학교 정류장 어디야?"',
            'route_detail': '"무학중학교 가는 방법 자세히 알려줘", "어느 출구로 나가야 해?", "어디서 내려야 해?"',
            'complex': '"무학중학교 가는데 시간 얼마나 걸려?", "무학중학교까지 가장 빠른 방법은?", "걸어갈 수도 있어?"'
        }
    else:  # english
        lang_instruction = "영어로"
        lang_note = "자연스러운 구어체 영어 (informal/conversational)"
        example_questions = {
            'basic': '"Where is Muhak Middle School?", "What is Muhak Middle School?", "How far is it?"',
            'transit': '"Do I need to take a bus?", "Which bus goes there?", "Where is the bus stop?"',
            'route_detail': '"Can you explain the route in detail?", "Which exit should I use?", "Where should I get off?"',
            'complex': '"How long does it take?", "What\'s the fastest way?", "Can I walk there?"'
        }
    
    # 배치 유형에 따른 프롬프트 설정 (대중교통 중심)
    prompts = {
        'basic': {
            'count': 8,
            'instruction': f"""
            다음 유형의 질문-답변 쌍을 {lang_instruction} 8개 생성해주세요:
            - 장소 이름/위치 질문 (3개)
            - 장소 특징/카테고리 질문 (2개)
            - 거리/접근성 질문 (3개)
            
            질문 스타일 예시: {example_questions['basic']}
            """
        },
        'transit': {
            'count': 12,
            'instruction': f"""
            다음 유형의 질문-답변 쌍을 {lang_instruction} 12개 생성해주세요:
            - 대중교통 이용 필요성 질문 (3개)
            - 버스 번호/노선 질문 (4개)
            - 정류장 위치 질문 (3개)
            - 출구/환승 정보 질문 (2개)
            
            질문 스타일 예시: {example_questions['transit']}
            """
        },
        'route_detail': {
            'count': 12,
            'instruction': f"""
            다음 유형의 질문-답변 쌍을 {lang_instruction} 12개 생성해주세요:
            - 상세 경로 안내 질문 (5개)
            - 하차 위치 질문 (3개)
            - 도보 시간 vs 버스 시간 비교 (2개)
            - 출구/탑승 위치 질문 (2개)
            
            질문 스타일 예시: {example_questions['route_detail']}
            """
        },
        'complex': {
            'count': 8,
            'instruction': f"""
            다음 유형의 질문-답변 쌍을 {lang_instruction} 8개 생성해주세요:
            - 복합 정보 질문 (장소 + 대중교통 경로) (3개)
            - 최적 경로/시간 질문 (2개)
            - 도보 vs 대중교통 비교 (2개)
            - 대체 경로 질문 (1개)
            
            질문 스타일 예시: {example_questions['complex']}
            """
        }
    }
    
    batch_config = prompts[batch_type]
    
    if language == "korean":
        prompt = f"""
당신은 한양대학교 주변 지역 대중교통 안내 전문가입니다.
아래 장소 정보를 바탕으로 학생들이 실제로 물어볼 법한 자연스러운 질문-답변 쌍을 한국어로 생성해주세요.

[장소 정보 - 1~2km 거리, 대중교통 권장]
```json
{location_str}
```

[생성 지침]
{batch_config['instruction']}

[중요 규칙]
1. 질문은 구어체로 자연스럽게 작성 ({lang_note})
2. 질문 표현을 최대한 다양하게:
   - 장소명 변형: "{location_info.get('name', '')}", 약칭이나 별칭도 활용
   - 질문 형식: "~어디야?", "~알려줘", "~어떻게 가?", "~버스 뭐 타?", "~정류장 어디?" 등
3. 답변은 친절하고 명확하게, 주어진 정보에만 근거
4. 답변 길이: 200-350자 내외 (대중교통 경로 설명)
5. 각 QA는 명확하게 구분되는 내용이어야 함 (중복 최소화)
6. 대중교통 정보를 정확히 포함:
   - 지하철 출구 번호
   - 버스 번호 (예: 성동03, 2012번)
   - 정류장 이름 (예: '한양대역' 정류장)
   - 하차 정류장
   ⚠️ 정류장 ID는 포함하지 마세요 (예: ID: 04-129 같은 정보 제외)
7. estimated_time_walking 정보를 활용 (도보 시간 vs 버스 시간 비교)
8. transport_type이 "대중교통 권장"임을 반영
9. 1~2km 거리이므로 "걸어가기엔 좀 멀어서 버스 추천" 뉘앙스 포함

[출력 형식]
반드시 다음 JSON 배열 형식으로만 출력하세요:

```json
[
  {{
    "question": "생성된 질문 1",
    "answer": "생성된 답변 1",
    "type": "{batch_type}",
    "location_name": "{location_info.get('name', '')}"
  }},
  {{
    "question": "생성된 질문 2",
    "answer": "생성된 답변 2",
    "type": "{batch_type}",
    "location_name": "{location_info.get('name', '')}"
  }}
]
```

정확히 {batch_config['count']}개의 QA 쌍을 생성해주세요.
"""
    else:  # english
        prompt = f"""
You are a public transportation guide expert for the Hanyang University area.
Based on the location information below, generate natural question-answer pairs in English that students would actually ask.

[Location Information - 1-2km distance, Public Transport Recommended]
```json
{location_str}
```

[Generation Guidelines]
{batch_config['instruction']}

[Important Rules]
1. Questions should be written in natural conversational style ({lang_note})
2. Vary question expressions as much as possible:
   - Location name variations: "{location_info.get('name', '')}", use abbreviations or common names
   - Question formats: "Where is~?", "Which bus~?", "How do I get~?", "Where's the bus stop~?", "What's the stop ID~?" etc.
3. Answers should be clear and friendly, based only on the given information
4. Answer length: 200-350 characters for public transport route descriptions
5. Each QA should be clearly distinct (minimize duplication)
6. Include accurate public transport information:
   - Subway exit numbers
   - Bus numbers (e.g., Seongdong 03, Bus 2012)
   - Bus stop names (e.g., 'Hanyang Univ. Station' stop)
   - Drop-off stop
   ⚠️ Do NOT include stop IDs (e.g., exclude information like ID: 04-129)
7. Use estimated_time_walking information (compare walking vs bus time)
8. Reflect that transport_type is "Public Transport Recommended"
9. Distance is 1-2km, so mention "a bit far to walk, bus recommended" nuance

[Output Format]
Output ONLY in the following JSON array format:

```json
[
  {{
    "question": "Generated question 1",
    "answer": "Generated answer 1",
    "type": "{batch_type}",
    "location_name": "{location_info.get('name', '')}"
  }},
  {{
    "question": "Generated question 2",
    "answer": "Generated answer 2",
    "type": "{batch_type}",
    "location_name": "{location_info.get('name', '')}"
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
                max_tokens=4000
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


def generate_qa_pairs_for_location(location_info, language="korean"):
    """
    단일 장소에 대해 40개의 QA 쌍을 생성합니다.
    배치별로 나눠서 생성하여 다양성을 확보합니다.
    """
    location_name = location_info.get('name', 'Unknown')
    lang_label = "🇰🇷" if language == "korean" else "🇺🇸"
    print(f"\n🚌 {lang_label} Processing: {location_name} ({language.upper()})")
    
    all_qa_pairs = []
    
    # 배치별 생성 (총 40개 목표) - 대중교통 중심
    batches = [
        ('basic', 1),         # 8개
        ('transit', 1),       # 12개 - 버스/정류장 정보
        ('route_detail', 1),  # 12개 - 상세 경로
        ('complex', 1),       # 8개
    ]
    
    for batch_type, batch_num in batches:
        qa_batch = generate_qa_batch(location_info, batch_type, batch_num, language)
        all_qa_pairs.extend(qa_batch)
        sleep(1)  # API 레이트 리밋 방지
    
    print(f"  📊 Total generated: {len(all_qa_pairs)} {language} QA pairs")
    return all_qa_pairs


def generate_all_qa_pairs(input_data):
    """
    모든 장소에 대해 한국어와 영어 QA 쌍을 생성합니다.
    """
    korean_qa_pairs = []
    english_qa_pairs = []
    korean_counter = 0
    english_counter = 0
    
    # 카테고리 한영 매핑
    category_map = {
        "교육": "Education",
        "상업시설": "Commercial",
        "의료": "Medical",
        "문화/체육": "Culture/Sports",
        "공공시설": "Public Facility",
        "주거": "Residential",
        "교통": "Transportation",
        "행정/부속건물": "Administrative/Auxiliary",
        "학술": "Academic",
        "기숙사": "Dormitory",
        "체육시설": "Sports Facility",
        "기타": "Others"
    }
    
    for location in tqdm(input_data, desc="Generating bilingual QA dataset (1-2km, Public Transport)"):
        try:
            # 한국어 QA 생성
            korean_pairs = generate_qa_pairs_for_location(location, language="korean")
            
            if korean_pairs:
                for qa in korean_pairs:
                    qa_pair = {
                        "id": f"KO_MID_{korean_counter:05d}",
                        "language": "korean",
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "type": qa.get("type", "unknown"),
                        "location_name": qa.get("location_name", ""),
                        "category": location.get("category", "")
                    }
                    korean_qa_pairs.append(qa_pair)
                    korean_counter += 1
            else:
                print(f"⚠️ Warning: No Korean QA pairs generated for {location.get('name', 'Unknown')}")
            
            sleep(2)  # 한영 생성 사이 대기
            
            # 영어 QA 생성
            english_pairs = generate_qa_pairs_for_location(location, language="english")
            
            english_category = category_map.get(location.get("category", ""), location.get("category", ""))
            
            if english_pairs:
                for qa in english_pairs:
                    qa_pair = {
                        "id": f"EN_MID_{english_counter:05d}",
                        "language": "english",
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "type": qa.get("type", "unknown"),
                        "location_name": qa.get("location_name", ""),
                        "category": english_category
                    }
                    english_qa_pairs.append(qa_pair)
                    english_counter += 1
            else:
                print(f"⚠️ Warning: No English QA pairs generated for {location.get('name', 'Unknown')}")
        
        except Exception as e:
            print(f"❌ Error processing {location.get('name', 'Unknown')}: {e}")
    
    return korean_qa_pairs, english_qa_pairs


def save_qa_pairs_to_json(qa_pairs, output_path, language):
    """생성된 QA 쌍 리스트를 JSON 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {language} QA pairs to: {output_path}")


def print_statistics(korean_qa, english_qa):
    """생성된 QA 데이터셋의 통계를 출력합니다."""
    print("\n" + "="*70)
    print("📊 MID-DISTANCE (1-2KM) BILINGUAL DATASET STATISTICS")
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
        print(f"  {qa_type:15s}: {count:4d} ({count/len(korean_qa)*100:.1f}%)")
    
    ko_location_counts = {}
    for qa in korean_qa:
        location = qa.get('location_name', 'Unknown')
        ko_location_counts[location] = ko_location_counts.get(location, 0) + 1
    
    print(f"\n[By Location]")
    if len(ko_location_counts) > 0:
        print(f"  Total locations: {len(ko_location_counts)}")
        print(f"  Avg QA per location: {len(korean_qa)/len(ko_location_counts):.1f}")
        print(f"  Min: {min(ko_location_counts.values())}, Max: {max(ko_location_counts.values())}")
    
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
        print(f"  {qa_type:15s}: {count:4d} ({count/len(english_qa)*100:.1f}%)")
    
    en_location_counts = {}
    for qa in english_qa:
        location = qa.get('location_name', 'Unknown')
        en_location_counts[location] = en_location_counts.get(location, 0) + 1
    
    print(f"\n[By Location]")
    if len(en_location_counts) > 0:
        print(f"  Total locations: {len(en_location_counts)}")
        print(f"  Avg QA per location: {len(english_qa)/len(en_location_counts):.1f}")
        print(f"  Min: {min(en_location_counts.values())}, Max: {max(en_location_counts.values())}")
    
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
    input_json_path = r"C:\Users\jaeyu\Desktop\gemma\LLM\hanyang_routes_gpt_1_2km.json"
    korean_output_path = r"C:\Users\jaeyu\Desktop\gemma\LLM\mid_distance_qa_korean.json"
    english_output_path = r"C:\Users\jaeyu\Desktop\gemma\LLM\mid_distance_qa_english.json"
    combined_output_path = r"C:\Users\jaeyu\Desktop\gemma\LLM\mid_distance_qa_combined.json"
    
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
        print("📝 SAMPLE QA PAIRS (1-2KM, PUBLIC TRANSPORT)")
        print("="*70)
        
        if korean_qa:
            print("\n🇰🇷 Korean Sample (First 2):")
            for i, qa in enumerate(korean_qa[:2], 1):
                print(f"\n[KO Sample {i}]")
                print(f"Q: {qa['question']}")
                print(f"A: {qa['answer'][:150]}..." if len(qa['answer']) > 150 else f"A: {qa['answer']}")
                print(f"Type: {qa['type']}, Location: {qa['location_name']}")
        
        if english_qa:
            print("\n🇺🇸 English Sample (First 2):")
            for i, qa in enumerate(english_qa[:2], 1):
                print(f"\n[EN Sample {i}]")
                print(f"Q: {qa['question']}")
                print(f"A: {qa['answer'][:150]}..." if len(qa['answer']) > 150 else f"A: {qa['answer']}")
                print(f"Type: {qa['type']}, Location: {qa['location_name']}")
        
        print("="*70)
        
        # 4. 통계 출력
        print_statistics(korean_qa, english_qa)
        
        # 5. JSON 파일로 저장 (분리)
        save_qa_pairs_to_json(korean_qa, korean_output_path, "Korean")
        save_qa_pairs_to_json(english_qa, english_output_path, "English")
        
        # 6. 통합 파일도 저장
        combined_qa = korean_qa + english_qa
        save_qa_pairs_to_json(combined_qa, combined_output_path, "Combined")
        
        print("\n" + "="*70)
        print("✅ Successfully generated MID-DISTANCE bilingual QA dataset!")
        print("="*70)
        print(f"🇰🇷 Korean QA pairs: {len(korean_qa)}")
        print(f"   Target: {len(input_data)} locations × 40 = {len(input_data) * 40}")
        print(f"   Achievement: {len(korean_qa)/(len(input_data)*40)*100:.1f}%")
        
        print(f"\n🇺🇸 English QA pairs: {len(english_qa)}")
        print(f"   Target: {len(input_data)} locations × 40 = {len(input_data) * 40}")
        print(f"   Achievement: {len(english_qa)/(len(input_data)*40)*100:.1f}%")
        
        print(f"\n🌍 Total QA pairs: {len(korean_qa) + len(english_qa)}")
        print(f"\n🚌 Focus: Bus routes, stop names, exit numbers (NO stop IDs)")
        print("="*70)

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_json_path}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()