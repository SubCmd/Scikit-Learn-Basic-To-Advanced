# Part 1: Machine Learning & Scikit-Learn 개요

## 1-1. 머신러닝이란/
: 머신러닝(Machine Learning)은 명시적으로 프로그래밍하지 않고, 데이터에서 패턴을 학습해 예측이나 결정을 내리는 알고리즘

- 전통적 프로그래밍과 차이?
> 전통적 프로그래밍: 입력 + 규칙 → 결과
> 머신러닝: 입력 + 결과 → 규칙(모델)


## 1-2. 머신러닝의 3가지 유형
1) 지도학습 (Supervised Learning)
> 정의: 정답(label)이 있는 데이터로 학습
> 목표: 새로운 입력에 대한 정답을 예측
> 분류(Classification): 카테고리 예측 (예: 스팸/정상 이메일)
> 회귀(Regression): 연속 값 예측 (예: 집값 예측)

2) 비지도학습 (Unsupervised Learning)
> 정의: 정닶 없이 데이터 구조/패턴을 발견
> 군집화(Clustering): 유사한 데이터끼리 그룹핑 (예: 고객 세그먼트)
> 차원 축소(Dimensionality Reduction): 특성 수를 줄이면서 정보 보존 (예: PCA)]

3) 강화학습 (Reinforcement Learning)
> 정의: 환경과 상호작용하며 보상을 최대화하는 방향으로 학습
> scikit-learn에서는 다루지 않음 (별도 라이브러리 사용)


## 1-3. Scikit-Learn이란?
> Python 가장 대표적인 머신러닝 라이브러리로, 다음 특징 가짐.
> 일관된 API 설계 (fit, predict, transform)
> 풍부한 내장 데이터셋과 전처리 도구
> 다양한 머신러닝 알고리즘 지원
> 우수한 문서화와 커뮤니티


## 1-4. 핵심 용어 정리
| 용어 | 영문 | 설명 |
|------|------|------|
| 특성 | Feature | 모델에 입력되는 독립 변수 (X) |
| 레이블/타겟 | Label/Target | 예측하려는 종속 변수 (y) |
| 샘플 | Sample | 하나의 데이터 행 |
| 훈련 세트 | Training Set | 모델 학습에 사용하는 데이터 |
| 테스트 세트 | Test Set | 모델 성능 평가에 사용하는 데이터 |
| 과적합 | Overfitting | 훈련 데이터에만 지나치게 잘 맞는 현상 |
| 과소적합 | Underfitting | 훈련 데이터도 제대로 학습하지 못하는 현상 |
| 하이퍼파라미터 | Hyperparameter | 사용자가 직접 설정하는 모델의 외부 설정값 |