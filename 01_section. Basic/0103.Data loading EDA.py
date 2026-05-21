# 코드 예시
# 2-1. 기본 EDA 템플릿

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def basic_eda(dataset, name="Dataset"):
    """scikit-learn 데이터셋에 대한 기본 EDA 수행"""

    # DataFrame 변환
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target

    print(f"{'='*60}")
    print(f"📊 {name} EDA Report")
    print(f"{'='*60}")

    # 1) Shape
    print(f"\n📐 Shape: {df.shape}")
    print(f"   샘플 수: {df.shape[0]}")
    print(f"   특성 수: {df.shape[1] - 1}")  # target 제외

    # 2) 데이터 타입
    print(f"\n📋 데이터 타입:")
    print(df.dtypes.value_counts().to_string())

    # 3) 결측치
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(f"\n✅ 결측치: 없음")
    else:
        print(f"\n⚠️ 결측치:")
        print(missing[missing > 0])

    # 4) 기초 통계량
    print(f"\n📈 기초 통계량:")
    print(df.describe().round(2).to_string())

    # 5) 타겟 분포
    print(f"\n🎯 타겟 분포:")
    target_counts = pd.Series(dataset.target).value_counts().sort_index()
    for idx, count in target_counts.items():
        name_label = dataset.target_names[idx] if hasattr(dataset, 'target_names') else idx
        ratio = count / len(dataset.target) * 100
        print(f"   {name_label}: {count}개 ({ratio:.1f}%)")

    return df

# 실행
iris = load_iris()
df = basic_eda(iris, "Iris")

"""
============================================================
📊 Iris EDA Report
============================================================

📐 Shape: (150, 5)
   샘플 수: 150
   특성 수: 4

📋 데이터 타입:
float64    4
int64      1

✅ 결측치: 없음

📈 기초 통계량:
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
count             150.00            150.00             150.00            150.00  150.00
mean                5.84              3.06               3.76              1.20    1.00
std                 0.83              0.44               1.77              0.76    0.82
min                 4.30              2.00               1.00              0.10    0.00
25%                 5.10              2.80               1.60              0.30    0.00
50%                 5.80              3.00               4.35              1.30    1.00
75%                 6.40              3.30               5.10              1.80    2.00
max                 7.90              4.40               6.90              2.50    2.00

🎯 타겟 분포:
   setosa: 50개 (33.3%)
   versicolor: 50개 (33.3%)
   virginica: 50개 (33.3%)
"""

# ================================================================================
# 2-2. 특성 간 상관관계 확인

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names) # columns 부분 명칭을 특정지어주는 코드
df['target'] = iris.target

# 상관계수 행렬
corr_matrix = df.corr()
print("=== 상관계수 행렬 ===")
print(corr_matrix.round(2))

# 타켓과 상관관계 (절댓값 기준 정렬)
target_corr = corr_matrix['target'].drop('target').abs().sort_values(ascending=False)
print("\n=== 타겟과의 상관관계 (절대값 내림차순) ===")
for feat, corr in target_corr.items():
    print(f"   {feat}: {corr:.4f}")

"""
=== 상관계수 행렬 ===
                   sepal length (cm)  sepal width (cm)  ...  petal width (cm)  target
sepal length (cm)               1.00             -0.12  ...              0.82    0.78
sepal width (cm)               -0.12              1.00  ...             -0.37   -0.43
petal length (cm)               0.87             -0.43  ...              0.96    0.95
petal width (cm)                0.82             -0.37  ...              1.00    0.96
target                          0.78             -0.43  ...              0.96    1.00

[5 rows x 5 columns]

=== 타겟과의 상관관계 (절대값 내림차순) ===
   petal width (cm): 0.9565
   petal length (cm): 0.9490
   sepal length (cm): 0.7826
   sepal width (cm): 0.4267
"""

# petal length와 petal width가 타겟과 가장 높은 상관관계
# → 이 두 특성이 분류에 가장 유용할 가능성이 높다

# ================================================================================
# 2-3. 시각화를 통한 EDA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# === 1) 히스토그램: 각 특성의 분포 확인 ===
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, col in enumerate(iris.feature_names):
    ax = axes[idx // 2, idx % 2]
    for class_idx, class_name in enumerate(iris.target_names):
        mask = df['target'] == class_idx
        ax.hist(df.loc[mask, col], alpha=0.6, label=class_name, bins=15)
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
plt.savefig("00_download\eda_histogram.png", dpi=150)
plt.close()

# === 2) 박스플롯: 이상치 확인 ===
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, col in enumerate(iris.feature_names):
    df.boxplot(column=col, by='target', ax=axes[idx])
    axes[idx].set_title(col)
plt.suptitle("Feature Distribution by Target Class", y=1.02)
plt.tight_layout()
plt.savefig("00_download\eda_boxplot.png", dpi=150)
plt.close()

# === 3) 산점도: 두 특성 간 관계 ===
fig, ax = plt.subplots(figsize=(8, 6))
for class_idx, class_name in enumerate(iris.target_names):
    mask = df['target'] == class_idx
    ax.scatter(
        df.loc[mask, 'petal length (cm)'],
        df.loc[mask, 'petal width (cm)'],
        label=class_name, alpha=0.7
    )
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.legend()
ax.set_title('Petal Length vs Petal Width')
plt.tight_layout()
plt.savefig("00_download\eda_scatter.png", dpi=150)
plt.close()


# ================================================================================
# 2-4. 회귀 데이터셋 EDA

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # 타겟: 중간 주택 가격 (만 달러)

print("=== 기초 통계량 ===")
print(df.describe().round(2))

# 타겟 분포 확인 (회귀에서는 value_counts 대신 통계량 분포)
print(f"\n=== 타겟(주택 가격) 분포 ===")
print(f"  평균: ${df['MedHouseVal'].mean():.2f}만")
print(f"  중앙값: ${df['MedHouseVal'].median():.2f}만")
print(f"  표준편차: ${df['MedHouseVal'].std():.2f}만")
print(f"  최소: ${df['MedHouseVal'].min():.2f}만")
print(f"  최대: ${df['MedHouseVal'].max():.2f}만")

# 특성별 스케일 차이 확인 (전처리 필요성 판단)
print(f"\n=== 특성별 범위 차이 (스케일링 필요성 판단) ===")
for col in housing.feature_names:
    range_val = df[col].max() - df[col].min()
    print(f"  {col:15s} | min: {df[col].min():10.2f} | max: {df[col].max():10.2f} | range: {range_val:10.2f}")

"""
=== 기초 통계량 ===
         MedInc  HouseAge  AveRooms  AveBedrms  ...  AveOccup  Latitude  Longitude  MedHouseVal
count  20640.00  20640.00  20640.00   20640.00  ...  20640.00  20640.00   20640.00     20640.00 
mean       3.87     28.64      5.43       1.10  ...      3.07     35.63    -119.57         2.07 
std        1.90     12.59      2.47       0.47  ...     10.39      2.14       2.00         1.15 
min        0.50      1.00      0.85       0.33  ...      0.69     32.54    -124.35         0.15 
25%        2.56     18.00      4.44       1.01  ...      2.43     33.93    -121.80         1.20 
50%        3.53     29.00      5.23       1.05  ...      2.82     34.26    -118.49         1.80 
75%        4.74     37.00      6.05       1.10  ...      3.28     37.71    -118.01         2.65 
max       15.00     52.00    141.91      34.07  ...   1243.33     41.95    -114.31         5.00 

[8 rows x 9 columns]

=== 타겟(주택 가격) 분포 ===
  평균: $2.07만
  중앙값: $1.80만
  표준편차: $1.15만
  최소: $0.15만
  최대: $5.00만

=== 특성별 범위 차이 (스케일링 필요성 판단) ===
  MedInc          | min:       0.50 | max:      15.00 | range:      14.50
  HouseAge        | min:       1.00 | max:      52.00 | range:      51.00
  AveRooms        | min:       0.85 | max:     141.91 | range:     141.06
  AveBedrms       | min:       0.33 | max:      34.07 | range:      33.73
  Population      | min:       3.00 | max:   35682.00 | range:   35679.00
  AveOccup        | min:       0.69 | max:    1243.33 | range:    1242.64
  Latitude        | min:      32.54 | max:      41.95 | range:       9.41
  Longitude       | min:    -124.35 | max:    -114.31 | range:      10.04
"""

# → MedInc(0~15)와 Population(3~35682)의 스케일이 크게 다름
# → 거리 기반 알고리즘(KNN, SVM) 사용 시 스케일링 필수

# ================================================================================
# Lv.1 기본 — EDA 체크리스트 수행

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
전체 데이터 Shape: (569, 31)
- 샐픔 수(행): 569개
- 특성 수(열): 30개 (Target 제외)
------------------------------------------------------------
<class 'pandas.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 31 columns):
 #   Column                   Non-Null Count  Dtype
---  ------                   --------------  -----
 0   mean radius              569 non-null    float64
 1   mean texture             569 non-null    float64
 2   mean perimeter           569 non-null    float64
 3   mean area                569 non-null    float64
 4   mean smoothness          569 non-null    float64
 5   mean compactness         569 non-null    float64
 6   mean concavity           569 non-null    float64
 7   mean concave points      569 non-null    float64
 8   mean symmetry            569 non-null    float64
 9   mean fractal dimension   569 non-null    float64
 10  radius error             569 non-null    float64
 11  texture error            569 non-null    float64
 12  perimeter error          569 non-null    float64
 13  area error               569 non-null    float64
 14  smoothness error         569 non-null    float64
 15  compactness error        569 non-null    float64
 16  concavity error          569 non-null    float64
 17  concave points error     569 non-null    float64
 18  symmetry error           569 non-null    float64
 19  fractal dimension error  569 non-null    float64
 20  worst radius             569 non-null    float64
 21  worst texture            569 non-null    float64
 22  worst perimeter          569 non-null    float64
 23  worst area               569 non-null    float64
 24  worst smoothness         569 non-null    float64
 25  worst compactness        569 non-null    float64
 26  worst concavity          569 non-null    float64
 27  worst concave points     569 non-null    float64
 28  worst symmetry           569 non-null    float64
 29  worst fractal dimension  569 non-null    float64
 30  target                   569 non-null    int64
dtypes: float64(30), int64(1)
memory usage: 137.9 KB
None
------------------------------------------------------------
0개 -> 결측치 없음
------------------------------------------------------------
                         count        mean         std  ...         50%          75%         max
mean radius              569.0   14.127292    3.524049  ...   13.370000    15.780000    28.11000
mean texture             569.0   19.289649    4.301036  ...   18.840000    21.800000    39.28000
mean perimeter           569.0   91.969033   24.298981  ...   86.240000   104.100000   188.50000
mean area                569.0  654.889104  351.914129  ...  551.100000   782.700000  2501.00000
mean smoothness          569.0    0.096360    0.014064  ...    0.095870     0.105300     0.16340
mean compactness         569.0    0.104341    0.052813  ...    0.092630     0.130400     0.34540
mean concavity           569.0    0.088799    0.079720  ...    0.061540     0.130700     0.42680
mean concave points      569.0    0.048919    0.038803  ...    0.033500     0.074000     0.20120
mean symmetry            569.0    0.181162    0.027414  ...    0.179200     0.195700     0.30400
mean fractal dimension   569.0    0.062798    0.007060  ...    0.061540     0.066120     0.09744
radius error             569.0    0.405172    0.277313  ...    0.324200     0.478900     2.87300
texture error            569.0    1.216853    0.551648  ...    1.108000     1.474000     4.88500
perimeter error          569.0    2.866059    2.021855  ...    2.287000     3.357000    21.98000
area error               569.0   40.337079   45.491006  ...   24.530000    45.190000   542.20000
smoothness error         569.0    0.007041    0.003003  ...    0.006380     0.008146     0.03113
compactness error        569.0    0.025478    0.017908  ...    0.020450     0.032450     0.13540
concavity error          569.0    0.031894    0.030186  ...    0.025890     0.042050     0.39600
concave points error     569.0    0.011796    0.006170  ...    0.010930     0.014710     0.05279
symmetry error           569.0    0.020542    0.008266  ...    0.018730     0.023480     0.07895
fractal dimension error  569.0    0.003795    0.002646  ...    0.003187     0.004558     0.02984
worst radius             569.0   16.269190    4.833242  ...   14.970000    18.790000    36.04000
worst texture            569.0   25.677223    6.146258  ...   25.410000    29.720000    49.54000
worst perimeter          569.0  107.261213   33.602542  ...   97.660000   125.400000   251.20000
worst area               569.0  880.583128  569.356993  ...  686.500000  1084.000000  4254.00000
worst smoothness         569.0    0.132369    0.022832  ...    0.131300     0.146000     0.22260
worst compactness        569.0    0.254265    0.157336  ...    0.211900     0.339100     1.05800
worst concavity          569.0    0.272188    0.208624  ...    0.226700     0.382900     1.25200
worst concave points     569.0    0.114606    0.065732  ...    0.099930     0.161400     0.29100
worst symmetry           569.0    0.290076    0.061867  ...    0.282200     0.317900     0.66380
worst fractal dimension  569.0    0.083946    0.018061  ...    0.080040     0.092080     0.20750
target                   569.0    0.627417    0.483918  ...    1.000000     1.000000     1.00000

[31 rows x 8 columns]
------------------------------------------------------------
클래스 0 (malignant): 212개 (37.26%)
클래스 1 (benign): 357개 (62.74%)
"""


# ================================================================================
# Lv.2 응용 — 특성 분석 및 상관관계





