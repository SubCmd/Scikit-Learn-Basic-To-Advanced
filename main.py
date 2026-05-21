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
# load_breast_cancer() 데이터셋에 대해 EDA를 수행하라:

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 데이터셋 로드
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 분석을 위해 pandas DataFrame으로 변환
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y

# 1. shape을 확인하고, 샘플 수와 특성 수를 출력하라
print(f"전체 데이터 Shape: {df.shape}")
print(f"- 샐픔 수(행): {X.shape[0]}개")
print(f"- 특성 수(열): {X.shape[1]}개 (Target 제외)")
print("-" * 60)

# 2. 모든 특성의 데이터 타입을 확인하라
print(df.info())
print("-" * 60)

# 3. 결측치가 있는지 확인하라
print(f"{df.isnull().sum().sum()}개 -> 결측치 없음")
print("-" * 60)

# 4. describe()로 기초 통계량을 확인하라
print(df.describe().T)
print("-" * 60)

# 5. 타겟(악성/양성) 클래스별 샘플 수와 비율을 출력하라
# 6. 이 데이터셋은 균형/불균형 중 어디에 해당하는지 판단하라

class_counts = df['target'].value_counts()
class_proportions = df['target'].value_counts(normalize=True)

for cls in sorted(class_counts.index):
    class_name = cancer.target_names[cls]
    count = class_counts[cls]
    proportion = class_proportions[cls] * 100
    print(f"클래스 {cls} ({class_name}): {count}개 ({proportion:.2f}%)")

"""

"""