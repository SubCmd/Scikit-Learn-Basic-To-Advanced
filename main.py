# matplotlib 한국어 깨짐 해결
from bdb import effective

from pandas.core.apply import reconstruct_func
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
rcParams['axes.unicode_minus'] = False     # 마이너스 부호 깨짐 방지

'''
rcParams['font.family'] = 'AppleGothic' # mac version
rcParams['axes.unicode_minus'] = False
'''

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