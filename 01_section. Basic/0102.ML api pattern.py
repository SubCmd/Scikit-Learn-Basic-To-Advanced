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




# ================================================================================
# Lv.2 응용 — Transformer 패턴 + Data Leakage 이해

# [과제]
# 다음 두 가지 방식으로 StandardScaler를 적용하고 결과를 비교하라:

