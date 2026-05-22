# 코드 예시
# 2-1. 기본 train_test_split

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

# 기본 원리 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 테스트 세트 20%
    random_state=42      # 재현성
)

print(f"전체: {X.shape[0]}개")
print(f"훈련: {X_train.shape[0]}개 ({X_train.shape[0]/X.shape[0]*100:.0f})%")
print(f"테스트: {X_test.shape[0]}개 ({X_test.shape[0]/X.shape[0]*100:.0f})%")

# 전체: 150개
# 훈련: 120개 (80)%
# 테스트: 30개 (20)%


# ================================================================================
# 2-2. stratify 사용 (분류 문제 필수)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

# === stratify 미사용 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=== stratify 미사용 ===")
print(f"원본 클래스 비율: {np.bincount(y) / len(y)}")
print(f"훈련 클래스 비율: {np.bincount(y_train) / len(y_train)}")
print(f"테스트 클래스 비율: {np.bincount(y_test) / len(y_test)}")

# === stratify 사용 ===
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== stratify=y 사용 ===")
print(f"원본 클래스 비율: {np.bincount(y) / len(y)}")
print(f"훈련 클래스 비율: {np.bincount(y_train_s) / len(y_train_s)}")
print(f"테스트 클래스 비율: {np.bincount(y_test_s) / len(y_test_s)}")

# === stratify 미사용 ===
# 원본 클래스 비율: [0.33333333 0.33333333 0.33333333]
# 훈련 클래스 비율: [0.33333333 0.34166667 0.325     ]
# 테스트 클래스 비율: [0.33333333 0.3        0.36666667]

# === stratify=y 사용 ===
# 원본 클래스 비율: [0.33333333 0.33333333 0.33333333]
# 훈련 클래스 비율: [0.33333333 0.33333333 0.33333333]
# 테스트 클래스 비율: [0.33333333 0.33333333 0.33333333]


# ================================================================================
# 2-3. 불균형 데이터에서 stratify의 중요성

import numpy as np
from sklearn.model_selection import train_test_split

# 불균형 데이터 시뮬레이션 (이탈: 5%, 유지: 95%)
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 5)
y = np.zeros(n_samples, dtype=int)
y[:50] = 1  # 5%만 이탈(1)

print(f"전체 이탈 비율: {y.mean():.3f} ({sum(y)}명/{len(y)}명)")

# === stratify 미사용 — 운이 나쁘면 테스트에 이탈 고객이 거의 없을 수 있음 ===
results_no_stratify = []
for seed in range(10):
    _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    results_no_stratify.append(y_test.mean())
print(f"\nstratify 미사용 — 테스트 이탈 비율 (10회):")
print(f"  범위: {min(results_no_stratify):.3f} ~ {max(results_no_stratify):.3f}")

# === stratify 사용 — 항상 5% 유지 ===
results_stratify = []
for seed in range(10):
    _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    results_stratify.append(y_test.mean())
print(f"\nstratify=y 사용 — 테스트 이탈 비율 (10회):")
print(f"  범위: {min(results_stratify):.3f} ~ {max(results_stratify):.3f}")
# → 항상 0.050 (5%)으로 일정


# 전체 이탈 비율: 0.050 (50명/1000명)

# stratify 미사용 — 테스트 이탈 비율 (10회):
#   범위: 0.025 ~ 0.070

# stratify=y 사용 — 테스트 이탈 비율 (10회):
#   범위: 0.050 ~ 0.050


# ================================================================================
# 2-4. random_state 재현성 확인

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

# 같은 random_state → 같은 결과
X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"같은 seed → 같은 결과: {(X_train1 == X_train2).all()}") # True

# 다른 random_state → 다른 결과
X_train3, X_test3, _, _ = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"다른 seed → 다른 결과: {(X_train1 == X_train3).all()}")  # False

# 같은 seed → 같은 결과: True
# 다른 seed → 다른 결과: False


# ================================================================================
# 2-5. DataFrame과 함께 사용

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# DataFrame에서 X와 y 분리
X = df.drop('target', axis=1)   # 또는 df[iris.feature_names]
y = df['target']

# train_test_split은 DataFrame도 지원
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train type: {type(X_train)}")  # pandas DataFrame 유지!
print(f"X_train shape: {X_train.shape}")
print(f"\nX_train 처음 3행:")
print(X_train.head(3))

# 인덱스가 섞여 있는 것에 주의
print(f"\nX_train 인덱스: {X_train.index.tolist()[:10]}")
# [73, 18, 118, 78, ...] — 원본 인덱스가 유지됨

'''
X_train type: <class 'pandas.DataFrame'>
X_train shape: (120, 4)

X_train 처음 3행:
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)     
8                  4.4               2.9                1.4               0.2     
106                4.9               2.5                4.5               1.7     
76                 6.8               2.8                4.8               1.4     

X_train 인덱스: [8, 106, 76, 9, 89, 146, 94, 133, 135, 117]
'''


# ================================================================================
# Lv.1 기본 — 분리 및 비율 확인

# [과제]
# load_wine() 데이터셋을 사용하여:

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
X, y = wine.data, wine.target

# 1. 80:20으로 데이터를 분리하라 (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 훈련/테스트 세트의 샘플 수를 출력하라

print(f"훈련 클래스 비율: {np.bincount(y_train) / len(y_train)}")
print(f"테스트 클래스 비율: {np.bincount(y_test) / len(y_test)}")

# 3. stratify 미사용과 사용의 클래스 비율을 각각 출력하여 비교하라

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 클래스 비율: {np.bincount(y_train_s) / len(y_train_s)}")
print(f"테스트 클래스 비율: {np.bincount(y_test_s) / len(y_test_s)}")

# 4. test_size를 0.1, 0.2, 0.3으로 변경하며
#    각각의 훈련/테스트 크기를 출력하라

test_size = [0.1, 0.2, 0.3]

for size in test_size:
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=size, random_state=42, stratify=y
    )
    print(f"test_size: {size} / 훈련 세트: {X_train.shape[0]:>3} / 테스트 세트: {X_test.shape[0]:>3}")

"""
훈련 클래스 비율: [0.31690141 0.40140845 0.28169014]
테스트 클래스 비율: [0.38888889 0.38888889 0.22222222]
훈련 클래스 비율: [0.33098592 0.40140845 0.26760563]
테스트 클래스 비율: [0.33333333 0.38888889 0.27777778]
test_size: 0.1 / 훈련 세트: 160 / 테스트 세트:  18
test_size: 0.2 / 훈련 세트: 142 / 테스트 세트:  36
test_size: 0.3 / 훈련 세트: 124 / 테스트 세트:  54
"""


# ================================================================================
# Lv.2 응용 — 분리 비율이 모델 성능에 미치는 영향





