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