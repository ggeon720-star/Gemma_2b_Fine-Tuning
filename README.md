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
- Motivation (Why are you doing this?)

- What do you want to see at the end?

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
