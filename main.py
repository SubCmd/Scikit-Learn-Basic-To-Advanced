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
# seaborn의 타이타닉 데이터셋을 전처리하라:

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드
df = sns.load_dataset('titanic')

# 1. 결측치 현황을 확인하라 (각 열의 결측 수와 비율)
print("=== 원본 데이터 ===")
print(df.head(3))
print(f"\n=== 결측치 현황 및 비율 ===")
missing_df = pd.DataFrame({
    '결측치 수': df.isnull().sum(),
    '결측 비율(%)': (df.isnull().sum() / len(df) * 100).round(1)
})
print(missing_df)

# 2. 다음 특성을 선택하라:
#    - 수치형: age, fare, sibsp, parch
#    - 범주형: sex, pclass (문자열로 변환), embarked

# pclass를 범주형으로 변환
df['pclass'] = df['pclass'].astype(str)

# 예측할 타겟 변수(생존 여부)
y = df['survived']

# 사용할 특성 정의
numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['sex', 'pclass', 'embarked']

X = df[numeric_features + categorical_features]

# 데이터 분할 (stratify=y 적용)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. ColumnTransformer를 구성하라:
#    - 수치형: SimpleImputer(median) → StandardScaler
#    - 범주형: SimpleImputer(most_frequent) → OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer로 결합
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 4. 전처리 후 shape과 특성 이름을 확인하라
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

print("\n=== 전처리 후 결과 ===")
print(f"훈련 데이터 Shape: {X_train_scaled.shape}")
print(f"테스트 데이터 Shape: {X_test_scaled.shape}")

# 전처리된 특성(Feature) 이름 추출
feature_names = preprocessor.get_feature_names_out()
print("\n전처리 후 특성 이름:")
print(list(feature_names))

# 5. 전처리된 데이터로 KNN(K=5)의 정확도를 측정하라

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== 모델 평가 ===")
print(f"KNN (K=5) 검증 정확도: {accuracy:.4f}")

'''
=== 원본 데이터 ===
   survived  pclass     sex   age  sibsp  ...  adult_male  deck  embark_town alive  alone       
0         0       3    male  22.0      1  ...        True   NaN  Southampton    no  False       
1         1       1  female  38.0      1  ...       False     C    Cherbourg   yes  False       
2         1       3  female  26.0      0  ...       False   NaN  Southampton   yes   True       

[3 rows x 15 columns]

=== 결측치 현황 및 비율 ===
             결측치 수  결측 비율(%)
survived         0       0.0
pclass           0       0.0
sex              0       0.0
age            177      19.9
sibsp            0       0.0
parch            0       0.0
fare             0       0.0
embarked         2       0.2
class            0       0.0
who              0       0.0
adult_male       0       0.0
deck           688      77.2
embark_town      2       0.2
alive            0       0.0
alone            0       0.0

=== 전처리 후 결과 ===
훈련 데이터 Shape: (712, 12)
테스트 데이터 Shape: (179, 12)

전처리 후 특성 이름:
['num__age', 'num__fare', 'num__sibsp', 'num__parch', 'cat__sex_female', 'cat__sex_male', 'cat__pclass_1', 'cat__pclass_2', 'cat__pclass_3', 'cat__embarked_C', 'cat__embarked_Q', 'cat__embarked_S']

=== 모델 평가 ===
KNN (K=5) 검증 정확도: 0.8212
'''


# [힌트]
# - df['pclass'] = df['pclass'].astype(str) — pclass를 범주형으로 변환
# - embarked에 결측치가 있으므로 SimpleImputer 필수
# - stratify=y 사용