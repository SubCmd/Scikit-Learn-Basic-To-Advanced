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