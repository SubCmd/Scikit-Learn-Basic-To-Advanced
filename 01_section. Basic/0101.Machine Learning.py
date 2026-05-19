# 코드 예시
# 2-0. numpy, pandas, scikit-learn 버전 확인

import sklearn
import numpy as np
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")


# ================================================================================
# 2-1. scikit-learn 내장 데이터셋 살펴보기

from sklearn.datasets import load_iris, fetch_california_housing

# === 분류용 데이터셋: Iris (붓꽃) ===
iris = load_iris()

# 데이터 구조 확인
print(type(iris))                    # sklearn.utils.Bunch (딕셔너리 유사 객체)
print(iris.keys())                   # dict_keys(['data', 'target', 'frame', ...])
print(f"특성(X) shape: {iris.data.shape}")      # (150, 4) — 150개 샘플, 4개 특성
print(f"타겟(y) shape: {iris.target.shape}")     # (150,) — 150개 레이블
print(f"특성 이름: {iris.feature_names}")         # ['sepal length', 'sepal width', ...]
print(f"타겟 클래스: {iris.target_names}")        # ['setosa', 'versicolor', 'virginica']

# === 회귀용 데이터셋: California Housing ===
housing = fetch_california_housing()
print(f"특성(X) shape: {housing.data.shape}")    # (20640, 8) — 20640개 샘플, 8개 특성
print(f"타겟(y) shape: {housing.target.shape}")  # (20640,) — 주택 가격 (연속값)
print(f"특성 이름: {housing.feature_names}")

"""
<class 'sklearn.utils._bunch.Bunch'>
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
특성(X) shape: (150, 4)
타겟(y) shape: (150,)
특성 이름: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']       
타겟 클래스: ['setosa' 'versicolor' 'virginica']
특성(X) shape: (20640, 8)
타겟(y) shape: (20640,)
특성 이름: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
"""

# ================================================================================
# 2-2. DataFrame으로 변환하여 탐색 (PM/분석가 친숙한 방식)

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

# DataFrame으로 변환
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

# 기본 탐색
print("=== shape ===")
print(df.shape)                   # (150, 6)

print("\n=== 처음 5행 ===")
print(df.head())

print("\n=== 기초 통계량 ===")
print(df.describe())

print("\n=== 결측치 확인 ===")
print(df.isnull().sum())

print("\n=== 클래스 분포 확인 ===")
print(df['target_name'].value_counts())

"""
=== shape ===
(150, 6)

=== 처음 5행 ===
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target target_name     
0                5.1               3.5                1.4               0.2       0      setosa     
1                4.9               3.0                1.4               0.2       0      setosa     
2                4.7               3.2                1.3               0.2       0      setosa     
3                4.6               3.1                1.5               0.2       0      setosa     
4                5.0               3.6                1.4               0.2       0      setosa     

=== 기초 통계량 ===
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)      target
count         150.000000        150.000000         150.000000        150.000000  150.000000
mean            5.843333          3.057333           3.758000          1.199333    1.000000
std             0.828066          0.435866           1.765298          0.762238    0.819232
min             4.300000          2.000000           1.000000          0.100000    0.000000
25%             5.100000          2.800000           1.600000          0.300000    0.000000
50%             5.800000          3.000000           4.350000          1.300000    1.000000
75%             6.400000          3.300000           5.100000          1.800000    2.000000
max             7.900000          4.400000           6.900000          2.500000    2.000000

=== 결측치 확인 ===
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
target               0
target_name          0
dtype: int64

=== 클래스 분포 확인 ===
target_name
setosa        50
versicolor    50
virginica     50
Name: count, dtype: int64
"""

# ================================================================================
# 2-3. 첫 번째 머신러닝 모델 (전체 흐름 미리보기)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1) 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 2) 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) 모델 생성 & 학습
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)        # fit: 모델 학습

# 4) 예측
y_pred = model.predict(X_test)     # predict: 예측

# 5) 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")   # 정확도: 1.0000 (Iris는 단순한 데이터)

# 6) 새로운 데이터로 예측
import numpy as np
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # 새로운 꽃의 특성
prediction = model.predict(new_flower)
print(f"예측 클래스: {iris.target_names[prediction[0]]}")  # setosa

"""
정확도: 1.0000
예측 클래스: setosa
"""

# 핵심 포인트: 위 코드가 scikit-learn의 전체 워크플로우
# `데이터 → 분리 → 모델 생성 → fit → predict → 평가` 패턴은 어떤 알고리즘을 쓰든 동일


# ================================================================================
# Lv.1 기본 — 데이터셋 탐색

# [과제]
# scikit-learn의 load_wine() 데이터셋을 사용하여:

# [힌트]
# from sklearn.datasets import load_wine
# wine = load_wine()

import pandas as pd
from sklearn.datasets import load_wine
wine = load_wine()

# 1. 데이터를 로드하고 X(특성)와 y(타겟)의 shape을 출력하라
X, y = wine.data, wine.target

print(f"X(특성): {X.shape}")
print(f"y(타겟): {y.shape}")

# 2. 특성 이름(feature_names)과 타겟 클래스(target_names)를 출력하라
print(f"특성 이름(feature_names): {wine.feature_names}")
print(f"타겟 클래스(target_names): {wine.target_names}")

# 3. DataFrame으로 변환 후 describe()로 기초 통계량을 확인하라
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

print(df.describe().T)

# 4. 타겟 클래스별 샘플 수를 확인하라 (균형 데이터인지 판단)
class_counts = df['target'].value_counts().sort_index()

for idx, count in enumerate(class_counts):
    class_name = wine.target_names[idx]
    print(f"클래스 {idx} ({class_name}): {count}개")

"""
X(특성): (178, 13)
y(타겟): (178,)
특성 이름(feature_names): ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
타겟 클래스(target_names): ['class_0' 'class_1' 'class_2']
                              count        mean         std  ...      50%       75%      max
alcohol                       178.0   13.000618    0.811827  ...   13.050   13.6775    14.83        
malic_acid                    178.0    2.336348    1.117146  ...    1.865    3.0825     5.80        
ash                           178.0    2.366517    0.274344  ...    2.360    2.5575     3.23        
alcalinity_of_ash             178.0   19.494944    3.339564  ...   19.500   21.5000    30.00        
magnesium                     178.0   99.741573   14.282484  ...   98.000  107.0000   162.00        
total_phenols                 178.0    2.295112    0.625851  ...    2.355    2.8000     3.88        
flavanoids                    178.0    2.029270    0.998859  ...    2.135    2.8750     5.08        
nonflavanoid_phenols          178.0    0.361854    0.124453  ...    0.340    0.4375     0.66        
proanthocyanins               178.0    1.590899    0.572359  ...    1.555    1.9500     3.58        
color_intensity               178.0    5.058090    2.318286  ...    4.690    6.2000    13.00        
hue                           178.0    0.957449    0.228572  ...    0.965    1.1200     1.71        
od280/od315_of_diluted_wines  178.0    2.611685    0.709990  ...    2.780    3.1700     4.00        
proline                       178.0  746.893258  314.907474  ...  673.500  985.0000  1680.00        
target                        178.0    0.938202    0.775035  ...    1.000    2.0000     2.00        

[14 rows x 8 columns]
클래스 0 (class_0): 59개
클래스 1 (class_1): 71개
클래스 2 (class_2): 48개
"""

# ================================================================================
# Lv.2 응용 — 전체 워크플로우 따라하기

# [과제]
# load_wine() 데이터셋에 KNeighborsClassifier를 적용하여:

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target

# 1. 데이터를 80:20으로 분리하라 (random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터 크기(X_train): {X_train.shape}")
print(f"테스트 데이터 크기(X_test): {X_test.shape}")

# 2. KNN 모델을 K=5로 생성하고 학습시켜라

knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(X_train, y_train)

# 3. 테스트 데이터로 예측하고, 정확도를 출력하라

y_pred_5 = knn_5.predict(X_test)
acc_5 = accuracy_score(y_test, y_pred_5)

print(f"K=5 일 때의 테스트 데이터 예측 정확도: {acc_5:.4f}")

# 4. K값을 1, 3, 5, 7, 9로 변경하며 각각의 정확도를 비교하라

k_values = [1, 3, 5, 7, 9]
accuracy_results = {}

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # model.score() 방식을 사용하여 정확도 측정
    acc = model.score(X_test, y_test)
    accuracy_results[k] = acc
    print(f"K = {k} | 정확도: {acc:.4f}")

# 5. 가장 높은 정확도를 보이는 K값은 무엇인가?

best_k = max(accuracy_results, key=accuracy_results.get)
best_acc = accuracy_results[best_k]

print(f"가장 높은 정확도를 보이는 K값: K = {best_k}")
print(f"최고 정확도: {best_acc:.4f}")

"""
X(특성): (178, 13)
y(타겟): (178,)
특성 이름(feature_names): ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
타겟 클래스(target_names): ['class_0' 'class_1' 'class_2']
                              count        mean         std  ...      50%       75%      max
alcohol                       178.0   13.000618    0.811827  ...   13.050   13.6775    14.83        
malic_acid                    178.0    2.336348    1.117146  ...    1.865    3.0825     5.80        
ash                           178.0    2.366517    0.274344  ...    2.360    2.5575     3.23        
alcalinity_of_ash             178.0   19.494944    3.339564  ...   19.500   21.5000    30.00        
magnesium                     178.0   99.741573   14.282484  ...   98.000  107.0000   162.00        
total_phenols                 178.0    2.295112    0.625851  ...    2.355    2.8000     3.88        
flavanoids                    178.0    2.029270    0.998859  ...    2.135    2.8750     5.08        
nonflavanoid_phenols          178.0    0.361854    0.124453  ...    0.340    0.4375     0.66        
proanthocyanins               178.0    1.590899    0.572359  ...    1.555    1.9500     3.58        
color_intensity               178.0    5.058090    2.318286  ...    4.690    6.2000    13.00        
hue                           178.0    0.957449    0.228572  ...    0.965    1.1200     1.71        
od280/od315_of_diluted_wines  178.0    2.611685    0.709990  ...    2.780    3.1700     4.00        
proline                       178.0  746.893258  314.907474  ...  673.500  985.0000  1680.00        
target                        178.0    0.938202    0.775035  ...    1.000    2.0000     2.00        

[14 rows x 8 columns]
클래스 0 (class_0): 59개
클래스 1 (class_1): 71개
클래스 2 (class_2): 48개

(myenv) C:\Users\ST-USER\Scikit-Learn-Basic-To-Advanced>python main.py
학습 데이터 크기(X_train): (142, 13)
테스트 데이터 크기(X_test): (36, 13)
K=5 일 때의 테스트 데이터 예측 정확도: 0.8056
K = 1 | 정확도: 0.7778
K = 3 | 정확도: 0.7500
K = 5 | 정확도: 0.8056
K = 7 | 정확도: 0.7222
K = 9 | 정확도: 0.8056
가장 높은 정확도를 보이는 K값: K = 5
최고 정확도: 0.8056
"""