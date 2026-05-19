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