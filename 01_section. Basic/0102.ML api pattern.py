# 코드 예시
# 2-1. Estimator 패턴 - 여러 알고리즘 동일 패턴 적용

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터 준비 (공통)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# === 3가지 모델 모두 동일한 패턴! ===
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)              # 동일: fit()
    y_pred = model.predict(X_test)           # 동일: predict()
    acc = accuracy_score(y_test, y_pred)     # 동일: 평가
    print(f"{name:25s} → 정확도: {acc:.4f}")

"""
KNN                       → 정확도: 1.0000
Logistic Regression       → 정확도: 1.0000
Decision Tree             → 정확도: 1.0000
"""

# > **핵심**: 알고리즘만 바꿔 끼우면 된다. `fit()` → `predict()` 패턴은 절대 변하지 않는다.

# ================================================================================
# 2-2. Transformer 패턴 - 스케일링 예시

from sklearn.preprocessing import StandardScaler
import numpy as np

# 원본 데이터 (단위가 다른 특성들)
X_train = np.array([
    [25, 50000],    # 나이(세), 연봉(원)
    [30, 70000],
    [35, 60000],
    [40, 80000],
])
X_test = np.array([
    [28, 55000],
    [33, 75000],
])

# === Transformer 패턴 ===
scaler = StandardScaler()

# 1) fit: 훈련 데이터에서 평균, 표준편차 학습
scaler.fit(X_train)
print(f"학습된 평균: {scaler.mean_}")       # [32.5, 65000.]
print(f"학습된 스케일: {scaler.scale_}")    # [5.59, 11180.34]

# 2) transform: 학습된 통계값으로 변환
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)    # 테스트도 같은 기준으로!

# 3) inverse_transform: 원래 스케일로 복원
X_restored = scaler.inverse_transform(X_train_scaled)
print(f"\n복원된 X_train:\n{X_restored}")   # 원본과 동일

"""
학습된 평균: [3.25e+01 6.50e+04]
학습된 스케일: [5.59016994e+00 1.11803399e+04]

복원된 X_train:
[[2.5e+01 5.0e+04]
 [3.0e+01 7.0e+04]
 [3.5e+01 6.0e+04]
 [4.0e+01 8.0e+04]]
"""

# ================================================================================
# 2-3. fit_transform() vs fit() + transform()

from sklearn.preprocessing import StandardScaler
import numpy as np

X_train = np.array([[1, 2], [3, 4], [5, 6]])
X_test = np.array([[7, 8]])

scaler = StandardScaler()

# === 방법 1: fit_transform (훈련 데이터에만 사용!) ===
X_train_scaled_v1 = scaler.fit_transform(X_train)

# === 방법 2: fit + transform 분리 (결과 동일) ===
scaler2 = StandardScaler()
scaler2.fit(X_train)
X_train_scaled_v2 = scaler2.transform(X_train)

# 두 결과는 동일
print(np.allclose(X_train_scaled_v1, X_train_scaled_v2))  # True

# === 테스트 데이터는 반드시 transform만! ===
X_test_scaled = scaler.transform(X_test)   # ✅ 올바름
# X_test_scaled = scaler.fit_transform(X_test)  # ❌ Data Leakage!


# ================================================================================
# 2-4. 학습된 속성 확인 (trailing underscore)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Estimator의 학습된 속성 ===
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

lr = LinearRegression()
# print(lr.coef_)  # ❌ NotFittedError — fit() 전에는 접근 불가!
 
lr.fit(X, y)
print(f"학습된 계수 (coef_): {lr.coef_}")           # [2.] — 기울기
print(f"학습된 절편 (intercept_): {lr.intercept_}")  # 0.0 — y절편
print(f"특성 수 (n_features_in_): {lr.n_features_in_}")  # 1

# === Transformer의 학습된 속성 ===
scaler = StandardScaler()
scaler.fit(X)
print(f"\n학습된 평균 (mean_): {scaler.mean_}")    # [3.]
print(f"학습된 스케일 (scale_): {scaler.scale_}")  # [1.41421356]

"""
학습된 계수 (coef_): [2.]
학습된 절편 (intercept_): -1.7763568394002505e-15
특성 수 (n_features_in_): 1

학습된 평균 (mean_): [3.]
학습된 스케일 (scale_): [1.41421356]
"""

# ================================================================================
# 2-5. get_params()와 set_params() - 하이퍼파라미터 관리

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# 현재 하이퍼파라미터 확인
print(knn.get_params())
# {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski',
#  'n_neighbors': 5, 'p': 2, 'weights': 'uniform', ...}

# 하이퍼파라미터 변경
knn.set_params(n_neighbors=3, weights='distance')
print(f"변경 후 K: {knn.get_params()['n_neighbors']}")  # 3
print(f"변경 후 weights: {knn.get_params()['weights']}")  # distance

"""
{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None,'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
변경 후 K: 3
변경 후 weights: distance
"""


# ================================================================================
# Lv.1 기본 — Estimator 패턴 연습

# [과제]
# load_wine() 데이터셋에 대해:

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 0. 데이터 로드 및 분리
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. DecisionTreeClassifier(max_depth=3, random_state=42)를 생성하라
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# 2. 2. fit()으로 훈련 데이터를 학습시켜라
model.fit(X_train, y_train)
print(f"모델 학습 완료")

# 3-4. predict()로 테스트 데이터를 예측, score()로 정확도 확인
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"테스트 데이터 예측 정확도 (Accuracy): {accuracy:.4f}")

# 5. get_params()로 현재 하이퍼파라미터를 출력하라
params = model.get_params()
for param_name, param_value in params.items():
    print(f"{param_name}: {param_value}")

# 6. 학습 후 feature_importances_ 속성을 확인하고,
#    가장 중요한 특성 3개를 출력하라

importances = model.feature_importances_
feature_imp_list = list(zip(wine.feature_names, importances))
feature_imp_list.sort(key=lambda x: x[1], reverse=True)

print("전체 특성 중요도 순위:")
for name, imp in feature_imp_list:
    print(f"- {name}: {imp:.4f}")

print("\n[결론]: 가장 중요한 특성 3개")
for i in range(3):
    name, imp = feature_imp_list[i]
    print(f"Top {i+1}: {name} ({imp:.4f})")

"""
모델 학습 완료
테스트 데이터 예측 정확도 (Accuracy): 0.9444
ccp_alpha: 0.0
class_weight: None
criterion: gini
max_depth: 3
max_features: None
max_leaf_nodes: None
min_impurity_decrease: 0.0
min_samples_leaf: 1
min_samples_split: 2
min_weight_fraction_leaf: 0.0
monotonic_cst: None
random_state: 42
splitter: best
전체 특성 중요도 순위:
- flavanoids: 0.4190
- color_intensity: 0.3924
- proline: 0.1673
- ash: 0.0213
- alcohol: 0.0000
- malic_acid: 0.0000
- alcalinity_of_ash: 0.0000
- magnesium: 0.0000
- total_phenols: 0.0000
- nonflavanoid_phenols: 0.0000
- proanthocyanins: 0.0000
- hue: 0.0000
- od280/od315_of_diluted_wines: 0.0000

[결론]: 가장 중요한 특성 3개
Top 1: flavanoids (0.4190)
Top 2: color_intensity (0.3924)
Top 3: proline (0.1673)
"""


# ================================================================================
# Lv.2 응용 — Transformer 패턴 + Data Leakage 이해

# [과제]
# 다음 두 가지 방식으로 StandardScaler를 적용하고 결과를 비교하라:

# 방식 A (올바른 방식):
#   1. X_train에만 fit
#   2. X_train과 X_test 각각 transform

# 방식 B (잘못된 방식 — Data Leakage):
#   1. X 전체(train+test)에 fit
#   2. X_train과 X_test 각각 transform

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