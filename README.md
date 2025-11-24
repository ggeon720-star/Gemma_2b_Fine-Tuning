# QLoRA를 이용한 Gemma-2B의 법률 특화 파인튜닝 
AIX 딥러닝 프로젝트

# Members
- 고재윤, (학부), (이메일)
- 권성근, (학부), (이메일)
- 신준희, 기계공학부, shinjh0331@naver.com
- 한인권, (학부), (이메일)

# Index
- 1. Proposal
  2. Datasets
  3. Methodology
  4. Evaluation & Analysis
  5. Related Work
  6. Conclusion: Discussion
  
# Proposal
Motivation (Why are you doing this?)
 해외에서는 LLM 기반 법률 서비스의 상용화가 빠르게 확산되고 있지만, 국내에서는 '데이터 접근성 부족, 개인정보보호법(PIPA)과 같은 규제 장벽, 법조계의 보수적 특성' 등의 이유로 더디게 확산되고 있습니다.
「강봉준 외 1명, 국내 법률 LLM의 활용과 연구동향 : 환각과 보안 리스크를 중심으로」

 특히 국내 법률 AI 도입 과정에서 환각 및 보안 리스크가 단순한 기술적 결함을 넘어 사회적 문제로 연결될 수 있음으로 정확도 이슈를 최소화해야 합니다.
 그렇기에 저희는 기존의 SLM 모델 (Gemma-2B)를 QLoRA를 활용하여 저비용으로 파인튜닝함으로서 더 전문적이고 문맥을 잘 이해하는 LLM을 만들고자 하셨습니다.

What do you want to see at the end?
1. 법률 Domain에서의 성능 향상
    - 파인튜닝한 모델의 성능 분석을 위한 평가 기준 필요
    - 기존 Gemma-2B과의 QA 정확도 비교
2. QLoRA (Quantized Low-Rank Adaptation)

# Datasets

# Methodology 
- Explaining your choice of algorithms (methods)
> 대략적인 알고리즘
> 1. 라이브러리 설치 및 드라이브 Mount
> 2. 판결문 파일 (JSON)을 정규화
> 3. 정규화된 파일들로 Gemma chat 프롬프트 텍스트 생성
> 4. QLoRA로 Gemma-2B 법률 모델 추가 학습
> 5. LoRA 어댑터 저장 & Base와 병합해 최종 모델 저장
> 6. 병합된 모델 4bit로 로드
- Explaining features (if any)

# Evaluation & Analysis
- Graphs, tables, any statistics (if any)

# Related Work (e.g., existing studies)
- Tools, libaries, blogs, or any documentatiton that you have used to do this project.

- Conclusion : Discussion
