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

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 0. 데이터 로드 및 80:20 분리
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# [실험]
# 1. 두 방식으로 각각 스케일링 후, KNN(K=5)을 학습시켜라

# ====================================================
# 방식 A (올바른 방식): X_train에만 fit
# ====================================================
scaler_A = StandardScaler()
# 1. X_train에만 fit
scaler_A.fit(X_train)
# 2. X_train과 X_test 각각 transform
X_train_A = scaler_A.transform(X_train)
X_test_A = scaler_A.transform(X_test)

# KNN 학습 몇 평가
knn_A = KNeighborsClassifier(n_neighbors=5)
knn_A.fit(X_train_A, y_train)
acc_A = knn_A.score(X_test_A, y_test)

# ====================================================
# 방식 B (잘못된 방식 - Data Leakage): 전체 X에 fit 
# ====================================================
scaler_B = StandardScaler()
# 1. X 전체(train + test)에 fit
scaler_B.fit(X)
# 2. X_train과 X_test 각각 transform
X_train_B = scaler_B.transform(X_train)
X_test_B = scaler_B.transform(X_test)

# KNN 학습 및 평가
knn_B = KNeighborsClassifier(n_neighbors=5)
knn_B.fit(X_train_B, y_train)
acc_B = knn_B.score(X_test_B, y_test)

# 2. 테스트 정확도를 비교하라
print(f"방식 A (올바른 방식) 정확도 : {acc_A:.4f}")
print(f"방식 B (데이터 누수) 정확도 : {acc_B:.4f}")
print("-" * 60)

# 3. scaler.mean_과 scaler.scale_이 두 방식에서 어떻게 다른지 출력하라
print("[방식 A - 학습 데이터로만 계산]")
print(f"평균(mean_): {scaler_A.mean_[:3]}")
print(f"표준편차(scale_): {scaler_A.scale_[:3]}")

print("\n[방식 B - 전체 데이터로 계산]")
print(f"평균(mean_): {scaler_B.mean_[:3]}")
print(f"표준편차(scale_): {scaler_B.scale_[:3]}")
print("-" * 60)

# 4. 왜 방식 B가 문제인지 1~2문장으로 설명하라
# 방식 B는 모델이 학습 중에 결코 알아서는 안 될 "미래 데이터(테스트셋)"의 평균과 표준편차  정보를 미리 반영(Data Leakage)하기 때문
# 이로 인해 테스트셋에 대한 정확도가 왜곡되어 높게 나올 수 있으며, 실제 서비스 환경에 배포했을 때, 성능이 크게 떨어지는 문제를 유발

"""
방식 A (올바른 방식) 정확도 : 0.9474
방식 B (데이터 누수) 정확도 : 0.9474
------------------------------------------------------------
[방식 A - 학습 데이터로만 계산]
평균(mean_): [14.11763516 19.18503297 91.88224176]
표준편차(scale_): [ 3.53192761  4.26131404 24.29528447]

[방식 B - 전체 데이터로 계산]
평균(mean_): [14.12729174 19.28964851 91.96903339]
표준편차(scale_): [ 3.52095076  4.29725464 24.27761929]
------------------------------------------------------------
"""