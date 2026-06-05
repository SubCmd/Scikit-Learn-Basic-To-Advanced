# 코드 예시
# 2-1. 기본 로지스틱 회귀

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
log_reg = LogisticRegression(C=1.0, max_iter=200, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = log_reg.predict(X_test_scaled)
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=cancer.target_names)}")

'''
정확도: 0.9825

              precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
'''


# ================================================================================
# 2-2. 확률 예측 (predict_proba)

# 확률 예측
y_prob = log_reg.predict_proba(X_test_scaled)
print(f"확률 shape: {y_prob.shape}")
print(f"클래스: {log_reg.classes_}")

# 첫 5개 샘플의 확률
for i in range(5):
    print(f"샘플 {i+1} 확률: {y_prob[i][0]:.4f} (양성), {y_prob[i][1]:.4f} (음성) → 예측: {y_pred[i]}")

'''
확률 shape: (114, 2)
클래스: [0 1]
샘플 1 확률: 1.0000 (양성), 0.0000 (음성) → 예측: 0
샘플 2 확률: 0.0000 (양성), 1.0000 (음성) → 예측: 1
샘플 3 확률: 0.9936 (양성), 0.0064 (음성) → 예측: 0
샘플 4 확률: 0.4665 (양성), 0.5335 (음성) → 예측: 1
샘플 5 확률: 1.0000 (양성), 0.0000 (음성) → 예측: 0
'''

# ================================================================================
# 2-3. 계수(가중치) 해석

import pandas as pd
import numpy as np

# 각 특성의 가중치 확인
coef_df = pd.DataFrame({
    'feature': cancer.feature_names,
    'coefficient': log_reg.coef_[0],
    'abs_coefficient': np.abs(log_reg.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("=== 특성별 계수 (절댓값 내림차순 Top 10) ===")
for _, row in coef_df.head(10).iterrows():
    direction = "양성↑" if row['coefficient'] > 0 else "악성↑"
    print(f"  {row['feature']:30s} | coef: {row['coefficient']:+.4f} | {direction}")

print(f"\n절편(intercept): {log_reg.intercept_[0]:.4f}")

# 해석 예시:
# worst radius coef=-1.23 → worst radius가 클수록 악성(0) 확률 증가
# worst concave points coef=-0.89 → 이 특성이 클수록 악성 확률 증가

'''
=== 특성별 계수 (절댓값 내림차순 Top 10) ===
  worst texture                  | coef: -1.2551 | 악성↑
  radius error                   | coef: -1.0830 | 악성↑
  worst concave points           | coef: -0.9537 | 악성↑
  worst area                     | coef: -0.9478 | 악성↑
  worst radius                   | coef: -0.9476 | 악성↑
  worst symmetry                 | coef: -0.9392 | 악성↑
  area error                     | coef: -0.9291 | 악성↑
  worst concavity                | coef: -0.8232 | 악성↑
  worst perimeter                | coef: -0.7632 | 악성↑
  worst smoothness               | coef: -0.7466 | 악성↑
'''


# ================================================================================
# 2-4.  규제 강도 C 비교

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100]

print(f"{'C':>8} | {'CV 평균':>8} | {'CV 표준편차':>10} | {'비제로 계수':>10}")
print("-" * 50)

for C in C_values:
    lr = LogisticRegression(C=C, max_iter=500, random_state=42)
    scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='accuracy')
    lr.fit(X_train_scaled, y_train)
    n_nonzero = np.sum(np.abs(lr.coef_[0]) > 0.01)
    print(f"{C:>8} | {scores.mean():>8.4f} | {scores.std():>10.4f} | {n_nonzero:>10}")

'''
       C |    CV 평균 |    CV 표준편차 |     비제로 계수
--------------------------------------------------
   0.001 |   0.8791 |     0.0251 |         27
    0.01 |   0.9516 |     0.0204 |         30
     0.1 |   0.9802 |     0.0162 |         29
     1.0 |   0.9802 |     0.0128 |         30
    10.0 |   0.9692 |     0.0162 |         30
     100 |   0.9560 |     0.0197 |         30
'''

# 비제로 계수란?
# 로지스틱 회귀에서 계수가 0이 되지 않은 값들

# L1 규제(Lasso)
# : 중요하지 않은 특성의 계수를 완전히 0으로 만듦
# 절대값의 합 / 특성 선택
# 용도 : 중요한 변수가 무엇인지?

# L2 규제(Ridge)
# : 중요하지 않은 특성의 계수를 0에 가깝게 작게 만듦
# 제곱의 합 / 변수 간 상관관계가 높을 때 안정적
# 용도 : 모든 변수를 유지하되 과대적합만 막고 싶을 때

# ================================================================================
# 2-5. 다중 클래스 분류

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 다중 클래스 로지스틱 회귀
log_reg = LogisticRegression(max_iter=200, random_state=42)
log_reg.fit(X_train_s, y_train)

# 3클래스 확률 예측
y_prob = log_reg.predict_proba(X_test_s)
print(f"확률 shape: {y_prob.shape}")  # (30, 3)
print(f"\n첫 번째 샘플:")
for cls, prob in zip(iris.target_names, y_prob[0]):
    print(f"  {cls}: {prob:.4f}")

# 계수: 클래스별로 각각 존재
print(f"\ncoef_ shape: {log_reg.coef_.shape}")  # (3, 4) — 3클래스 × 4특성

'''
확률 shape: (30, 3)

첫 번째 샘플:
  setosa: 0.9788
  versicolor: 0.0212
  virginica: 0.0000

coef_ shape: (3, 4)
'''

# Scikit-learn 1.2버전부터는 multi_class 매개변수를 사용하지 않음.


# ================================================================================
# Lv.1 기본 — 결측치 처리 연습

# [과제]
# load_breast_cancer() 데이터에 대해:

# 1. 로지스틱 회귀 모델을 학습시키고 정확도를 출력하라
# 2. 테스트 데이터 중 예측이 틀린 샘플의 인덱스를 찾아라
# 3. 틀린 샘플들의 predict_proba() 결과를 확인하라
#    → 확률이 0.5 근처인가? (경계선 근처의 애매한 샘플인지 확인)
# 4. 가장 영향력 있는 특성 5개를 계수 절대값 기준으로 출력하라

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
log_reg = LogisticRegression(C=1.0, max_iter=200, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 예측
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1] # 양성(1) 확률만 추출

# 정확도
print(f"테스트 데이터 정확도: {accuracy_score(y_test, y_pred):.4f}")

# 틀린 샘플 확인
wrong_indices = np.where(y_pred != y_test)[0]
print(f"틀린 샘플 인덱스: {wrong_indices}")

if len(wrong_indices) > 0:
    first_wrong = wrong_indices[0]
    print(f"\n[첫 번째 틀린 샘플(인덱스 {first_wrong}) 분석]")
    print(f"실제 정답: {y_test[first_wrong]} | 모델 예측: {y_pred[first_wrong]}")
    print(f"양성(1)일 확률: {y_prob[first_wrong]:.4f} (0.5에 가까울수록 불확실)")

'''
테스트 데이터 정확도: 0.9825
틀린 샘플 인덱스: [16 53]

[첫 번째 틀린 샘플(인덱스 16) 분석]
실제 정답: 1 | 모델 예측: 0
양성(1)일 확률: 0.3659 (0.5에 가까울수록 불확실)
'''

# [힌트]
# - 틀린 인덱스: np.where(y_pred != y_test)[0]
# - 확률이 0.5 근처일수록 모델이 불확실한 샘플


# ================================================================================
# Lv.2 응용 — 타이타닉 데이터 전처리


