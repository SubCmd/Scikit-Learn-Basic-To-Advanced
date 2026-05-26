# 코드 예시
# 2-1. StandardScaler (가장 많이 사용)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

# StandardScaler 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 변환 전/후 비교
print("=== 변환 전 (X_train) ===")
print(f"  평균: {X_train.mean(axis=0).round(2)}")
print(f"  표준편차: {X_train.std(axis=0).round(2)}")

print("\n=== 변환 후 (X_train_scaled) ===")
print(f"  평균: {X_train_scaled.mean(axis=0).round(2)}")   # [0, 0, 0, 0]에 가까움
print(f"  표준편차: {X_train_scaled.std(axis=0).round(2)}") # [1, 1, 1, 1]에 가까움

# 학습된 속성 확인
print(f"\n학습된 평균 (mean_): {scaler.mean_.round(2)}")
print(f"학습된 스케일 (scale_): {scaler.scale_.round(2)}")

'''
=== 변환 전 (X_train) ===
  평균: [5.84 3.05 3.77 1.2 ]
  표준편차: [0.84 0.45 1.76 0.76]

=== 변환 후 (X_train_scaled) ===
  평균: [-0. -0.  0.  0.]
  표준편차: [1. 1. 1. 1.]

학습된 평균 (mean_): [5.84 3.05 3.77 1.2 ]
학습된 스케일 (scale_): [0.84 0.45 1.76 0.76]
'''


# ================================================================================
# 2-2. MinMaxScaler (0~1 범위)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)

print("=== MinMaxScaler 변환 후 ===")
print(f"  최소: {X_train_mm.min(axis=0)}")   # [0, 0, 0, 0]
print(f"  최대: {X_train_mm.max(axis=0)}")   # [1, 1, 1, 1]

# 커스텀 범위 설정 (예: -1 ~ 1)
scaler_custom = MinMaxScaler(feature_range=(-1, 1))
X_train_custom = scaler_custom.fit_transform(X_train)
print(f"\n커스텀 범위 (-1~1):")
print(f"  최소: {X_train_custom.min(axis=0)}")  # [-1, -1, -1, -1]
print(f"  최대: {X_train_custom.max(axis=0)}")  # [1, 1, 1, 1]

# ⚠️ 테스트 데이터는 0~1 범위를 벗어날 수 있음!
print(f"\n테스트 데이터 최소: {X_test_mm.min(axis=0).round(2)}")
print(f"테스트 데이터 최대: {X_test_mm.max(axis=0).round(2)}")
# → 훈련 데이터의 min/max로 변환하므로 범위를 벗어날 수 있음

'''
=== MinMaxScaler 변환 후 ===
  최소: [0. 0. 0. 0.]
  최대: [1. 1. 1. 1.]

커스텀 범위 (-1~1):
  최소: [-1. -1. -1. -1.]
  최대: [1. 1. 1. 1.]

테스트 데이터 최소: [ 0.03  0.12 -0.02  0.04]
테스트 데이터 최대: [0.83 0.83 0.9  0.96]
'''


# ================================================================================
# 2-3. RobustScaler (이상치에 강건)

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

# 이상치가 포함된 데이터 시뮬레이션
np.random.seed(42)
X_with_outliers = np.random.randn(100, 2) * 10 + 50
X_with_outliers[0] = [500, 500]     # 이상치 추가
X_with_outliers[1] = [-200, -200]   # 이상치 추가

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
}

for name, scaler in scalers.items():
    X_scaled = scaler.fit_transform(X_with_outliers)
    # 이상치를 제외한 정상 데이터(2번 인덱스부터)의 범위 확인
    normal_data = X_scaled[2:]
    print(f"{name:20s} | 정상 데이터 범위: ")
    print(f"{normal_data.min():.2f}, {normal_data.max():.2f}")

'''
StandardScaler       | 정상 데이터 범위: 
-0.52, 0.48
MinMaxScaler         | 정상 데이터 범위:
0.32, 0.40
RobustScaler         | 정상 데이터 범위:
-2.28, 1.87
'''

# StandardScaler | 정상 데이터 범위: [-0.52, 0.48]  ← 이상치 때문에 좁은 범위
# MinMaxScaler   | 정상 데이터 범위: [0.32, 0.40]    ← 대부분 0.3~0.4에 몰림!
# RobustScaler   | 정상 데이터 범위: [-2.28, 1.87]   ← 정상 데이터가 넓게 분포


# ================================================================================
# 2-4. 스케일링이 모델 성능에 미치는 영향

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터 준비
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)

# 스케일러 목록
scalers = {
    'None (원본)': None, 
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
}

# 모델 목록
models = {
    'KNN(K=5)': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42)
}

print(f"{'스케일러':<20} {'KNN(K=5)':>10} {'DecisionTree':>15}")
print("-" * 50)

for scaler_name, scaler in scalers.items():
    results = []
    for model_name, model in models.items():
        # 매번 새 모델 인스턴스 생성
        model_instance = model.__class__(**model.get_params())

        if scaler is not None:
            scaler_instance = scaler.__class__()
            X_tr = scaler_instance.fit_transform(X_train)
            X_te = scaler_instance.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test
        
        model_instance.fit(X_tr, y_train)
        acc = model_instance.score(X_te, y_test)
        results.append(acc)

    print(f"{scaler_name:<20} {results[0]:>10.4f} {results[1]:>15.4f}")

'''
스케일러                   KNN(K=5)    DecisionTree
--------------------------------------------------
None (원본)                0.8056          0.9444
StandardScaler           0.9722          0.9444  ← KNN 대폭 향상!
MinMaxScaler             1.0000          0.9444
RobustScaler             0.9722          0.9444
'''
# → KNN은 스케일링 효과가 극적, DecisionTree는 영향 없음


# ================================================================================
# 2-5. inverse_transofrm - 원래 스케일로 복원

import numpy as np
from sklearn.preprocessing import StandardScaler

# 원본 데이터
X_original = np.array([[25, 50000], [30, 70000], [35, 60000]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)

# 스케일링된 데이터를 원래 스케일로 복원
X_restored = scaler.inverse_transform(X_scaled)

print("원본:")
print(X_original)
print("\n스케일링:")
print(X_scaled.round(4))
print("\n복원:")
print(X_restored.round(4))
# → 원본과 동일

'''
원본:
[[   25 50000]
 [   30 70000]
 [   35 60000]]

스케일링:
[[-1.2247 -1.2247]
 [ 0.      1.2247]
 [ 1.2247  0.    ]]

복원:
[[2.5e+01 5.0e+04]
 [3.0e+01 7.0e+04]
 [3.5e+01 6.0e+04]]
'''

# 사용 예시: 예측 결과를 원래 단위로 복원
# model.predict()가 스케일된 값을 반환하는 경우
# scaler.inverse_transform(predictions)으로 원래 단위로 변환


# ================================================================================
# Lv.1 기본 — 3대 스케일러 적용 및 비교

# [과제]
# load_wine() 데이터셋에 대해:

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

wine = load_wine()
X, y = wine.data, wine.target

# 1. 훈련/테스트 분리 후 (80:20, stratify=y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=y
)

# 2. StandardScaler, MinMaxScaler, RobustScaler를 각각 적용하라

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
}

# 3. 각 스케일러 적용 후 X_train의 각 특성별 평균, 표준편차, 최소, 최대를 출력하라
summary_stats = []

for name, scaler in scalers.items():
    # 훈련 데이터에 fit 및 transform 적용
    X_train_scaled = scaler.fit_transform(X_train)

    # 각 특성(axis=0)별 통계별 계산
    means = X_train_scaled.mean(axis=0)
    stds = X_train_scaled.std(axis=0)
    mins = X_train_scaled.min(axis=0)
    maxs = X_train_scaled.max(axis=0)

    summary_stats.append({
    'Scaler': name,
    'Mean (평균)': np.round(means.mean(), 4),
    'Std (표준편차)': np.round(stds.mean(), 4),
    'Min (최솟값)': np.round(mins.mean(), 4),
    'MAX (최댓값)': np.round(maxs.mean(), 4),
})

# 4. 결과를 비교표(DataFrame)로 정리하라
df_summary = pd.DataFrame(summary_stats)
print(df_summary.to_string(index=False))

# [힌트]
# - X_train_scaled.mean(axis=0)으로 특성별 평균 계산
# - pd.DataFrame으로 결과를 정리하면 보기 편함

'''
        Scaler  Mean (평균)  Std (표준편차)  Min (최솟값)  MAX (최댓값)
StandardScaler     0.0000      1.0000    -2.0976     2.8763
  MinMaxScaler     0.4226      0.2080     0.0000     1.0000
  RobustScaler     0.0446      0.6978    -1.4316     2.0879
'''


# ================================================================================
# Lv.2 응용 — 스케일링 + 모델 성능 실험

# [과제]
# load_breast_cancer() 데이터셋에 대해:

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. 4가지 조건으로 KNN(K=5) 모델을 학습하라:
#   - 스케일링 없음
#   - StandardScaler
#   - MinMaxScaler
#   - RobustScaler

scalers = {
    'None (원본)': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
}

results = {scaler_name: [] for scaler_name in scalers.keys()}
k_range = range(1, 21)

# 2. 각 조건에서 K를 1~20까지 변경하며 테스트 정확도를 기록하라

for name, scaler in scalers.items():
    # 스케일링 적용 단계
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
# K값 1~20 순회 학습 및 평가
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        acc = knn.score(X_test_scaled, y_test)
        results[name].append(acc)    

# 3. 결과를 시각화하라:
#   - X축: K값, Y축: 정확도
#   - 4개의 선 (스케일러별)을 하나의 그래프에 표시
plt.figure(figsize=(12, 7))

markers = ['o', 's', '^', 'd']
for (name, acc_list), marker in zip(results.items(), markers):
    plt.plot(k_range, acc_list, label=name, marker=marker, linewidth=2)

plt.title('KNN Accuracy Comparison by Scaler and K value', fontsize=14, pad=15)
plt.xlabel('K Value (n_neighbors)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.show()

# 4. 스케일링 없이 최적 K와, StandardScaler 적용 후 최적 K를 비교하라.
#    차이가 나는 이유를 설명하라.
print("=" * 70)
print("## 4. 최적 K값 비교 및 결론")
print("=" * 70)

for name, acc_list in results.items():
    best_acc = max(acc_list)
    # 가장 높은 정확도를 가진 K값들을 리스트로 추출 (동점 포함)
    best_ks = [k_range[i] for i, acc in enumerate(acc_list) if acc == best_acc]
    print(f"- {name:<15} : 최고 정확도 {best_acc:.4f} (최적 K = {best_ks})")

# [힌트]
# - for문 2중 중첩: 외부(스케일러), 내부(K값)
# - plt.plot()으로 선 그래프
# - 결과를 dict에 저장하여 나중에 비교