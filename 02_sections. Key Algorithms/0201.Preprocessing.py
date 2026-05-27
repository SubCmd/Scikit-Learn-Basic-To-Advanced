# 코드 예시
# 2-1. 결측치 확인 및 SimpleImputer

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# 결측치가 있는 데이터 생성
data = {
    'age': [25, 30, np.nan, 40, 35, np.nan, 28],
    'income': [50000, np.nan, 60000, 80000, np.nan, 70000, 55000],
    'city': ['서울', '부산', '서울', np.nan, '대전', '부산', np.nan],
}
df = pd.DataFrame(data)

print("=== 원본 데이터 ===")
print(df)
print(f"\n=== 결측치 현황 ===")
print(df.isnull().sum())
print(f"\n결측 비율:")
print((df.isnull().sum() / len(df) * 100).round(1))

# === 수치형 결측치: 중앙값 대체 ===
imputer_num = SimpleImputer(strategy='median')
df[['age', 'income']] = imputer_num.fit_transform(df[['age', 'income']])

# === 범주형 결측치: 최빈값 대체 ===
imputer_cat = SimpleImputer(strategy='most_frequent')
df[['city']] = imputer_cat.fit_transform(df[['city']])

print("\n=== 결측치 처리 후 ===")
print(df)
print(f"\n결측치: {df.isnull().sum().sum()}개")  # 0개

'''
=== 원본 데이터 ===
    age   income city
0  25.0  50000.0   서울
1  30.0      NaN   부산
2   NaN  60000.0   서울
3  40.0  80000.0  NaN
4  35.0      NaN   대전
5   NaN  70000.0   부산
6  28.0  55000.0  NaN

=== 결측치 현황 ===
age       2
income    2
city      2
dtype: int64

결측 비율:
age       28.6
income    28.6
city      28.6
dtype: float64

=== 결측치 처리 후 ===
    age   income city
0  25.0  50000.0   서울
1  30.0  60000.0   부산
2  30.0  60000.0   서울
3  40.0  80000.0   부산
4  35.0  60000.0   대전
5  30.0  70000.0   부산
6  28.0  55000.0   부산

결측치: 0개
'''

# ================================================================================
# 2-2. SimpleImputer 전략별 비교

import numpy as np
from sklearn.impute import SimpleImputer

# 이상치 포함 데이터
X = np.array([[10], [20], [30], [40], [np.nan], [1000]])    # 1000은 이상치

strategies = {
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'constant(-1)': SimpleImputer(strategy='constant', fill_value=-1),
}

for name, imputer in strategies.items():
    X_filled = imputer.fit_transform(X)
    nan_idx = 4 # NaN이 있던 위치
    print(f"{name:15s} → NaN 대체값: {X_filled[nan_idx][0]:.1f}")

# mean            → NaN 대체값: 220.0   ← 이상치(1000)에 영향 받음!
# median          → NaN 대체값: 25.0    ← 이상치에 강건
# constant(-1)    → NaN 대체값: -1.0    ← 고정값


# ================================================================================
# 2-3. KNNImputer — 유사 샘플 기반 대체

import numpy as np
from sklearn.impute import KNNImputer

X = np.array([
    [1, 2, 3],
    [1, 3, 4],
    [1, np.nan, 3],    # 유사한 행(0, 1)의 값으로 추정
    [10, 20, 30],
    [10, 22, np.nan],  # 유사한 행(3)의 값으로 추정
])

imputer = KNNImputer(n_neighbors=2)
X_filled = imputer.fit_transform(X)

print("원본:")
print(X)
print("\nKNN 대체 후:")
print(X_filled)

'''
원본:
[[ 1.  2.  3.]
 [ 1.  3.  4.]
 [ 1. nan  3.] -> 가장 가까운 행 2개 : 1행& 2행
 [10. 20. 30.]
 [10. 22. nan]]

KNN 대체 후:
[[ 1.   2.   3. ]
 [ 1.   3.   4. ]
 [ 1.   2.5  3. ]
 [10.  20.  30. ]
 [10.  22.  16.5]]
'''

# [1, NaN, 3] → [1, 2.5, 3]  (행0의 2와 행1의 3의 평균)
# [10, 22, NaN] → [10, 22, 30]  (가장 유사한 행3의 값)


# ================================================================================
# 2-4. LabelEncoder — 타겟(y) 인코딩

from sklearn.preprocessing import LabelEncoder

# 타겟 레이블 인코딩 (y에만 사용!)
le = LabelEncoder()
y = ['이탈', '유지', '유지', '이탈', '유지']
y_encoded = le.fit_transform(y)

print(f"원본: {y}")
print(f"인코딩: {y_encoded}")         # [0, 1, 1, 0, 1]
print(f"클래스 매핑: {le.classes_}")  # ['이탈', '유지']

# 역변환
y_decoded = le.inverse_transform(y_encoded)
print(f"역변환: {y_decoded}")          # ['이탈', '유지', '유지', '이탈', '유지']

'''
원본: ['이탈', '유지', '유지', '이탈', '유지']
인코딩: [1 0 0 1 0]
클래스 매핑: ['유지' '이탈']
역변환: ['이탈' '유지' '유지' '이탈' '유지']
'''


# ================================================================================
# 2-5. OneHotEncoder — 특성(X) 인코딩

import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 순서 없는 범주형 특성
X_cat = np.array([['서울'], ['부산'], ['대전'], ['서울'], ['부산']])

# === drop=None (기본) ===
ohe = OneHotEncoder(sparse_output=False)
X_encoded = ohe.fit_transform(X_cat)
print("drop=None:")
print(f"  카테고리: {ohe.categories_}")
print(f"  변환 결과:\n{X_encoded}")
# 대전→[1,0,0], 부산→[0,1,0], 서울→[0,0,1]

# === drop='first' (다중공선성 방지) ===
ohe_drop = OneHotEncoder(sparse_output=False, drop='first')
X_encoded_drop = ohe_drop.fit_transform(X_cat)
print(f"\ndrop='first':")
print(f"  변환 결과:\n{X_encoded_drop}")
# 대전→[0,0], 부산→[1,0], 서울→[0,1]

# === handle_unknown='ignore' (테스트에 새로운 범주가 나올 때) ===
ohe_safe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_safe.fit(X_cat)
X_new = np.array([['인천']])  # 학습 시 없던 범주
X_new_encoded = ohe_safe.transform(X_new)
print(f"\n미지 범주 'ì¸ì²':")
print(f"  변환 결과: {X_new_encoded}")  # [0, 0, 0] — 모두 0

'''
drop=None:
  카테고리: [array(['대전', '부산', '서울'], dtype='<U2')]
  변환 결과:
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]]

drop='first':
  변환 결과:
[[0. 1.]
 [1. 0.]
 [0. 0.]
 [0. 1.]
 [1. 0.]]

미지 범주 'ì¸ì²':
  변환 결과: [[0. 0. 0.]]
'''


# ================================================================================
# 2-6. OrdinalEncoder — 순서형 특성 인코딩

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# 순서가 있는 범주 (low < medium < high)
X_ordinal = np.array([['medium'], ['high'], ['low']])

# 순서를 명시적으로 지정!
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_encoded = oe.fit_transform(X_ordinal)

print("순서형 인코딩:")
for original, encoded in zip(X_ordinal.flatten(), X_encoded.flatten()):
    print(f"  {original} → {int(encoded)}")

'''
순서형 인코딩:
  medium → 1
  high → 2
  low → 0
'''

# zip 내장객체 : 두 리스트를 튜플 형식으로 반환


# ================================================================================
# 2-7. ColumnTransformer — 실전 전처리 파이프라인

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 실전형 데이터 생성
data = {
    'age': [25, 30, np.nan, 40, 35, 28, np.nan, 33],
    'income': [50000, np.nan, 60000, 80000, np.nan, 55000, 70000, 65000],
    'gender': ['M', 'F', 'M', 'F', np.nan, 'M', 'F', 'M'],
    'plan': ['basic', 'premium', 'basic', np.nan, 'premium', 'basic', 'premium', 'basic'],
    'churned': [0, 0, 1, 0, 1, 0, 0, 1],  # 타겟: 이탈 여부
}
df = pd.DataFrame(data)

# 특성과 타겟 분리
X = df.drop('churned', axis=1)
y = df['churned']

# 수치형/범주형 컬럼 지정
numeric_features = ['age', 'income']
categorical_features = ['gender', 'plan']

# 수치형 파이프라인: 결측치(중앙값) → 스케일링
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 범주형 파이프라인: 결측치(최빈값) → 원핫인코딩
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])
# handle_unknown='ignore' : 나중에 테스트 데이터에서 학습할때 보지 못했던 새로운 범주가 들어와도 에러를 내지 않음.

# ColumnTransformer로 합치기
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# 전처리 실행
X_processed = preprocessor.fit_transform(X)

print(f"원본 shape: {X.shape}")           # (8, 4)
print(f"전처리 후 shape: {X_processed.shape}")  # (8, 6) — 원핫 때문에 증가

# 전처리된 특성 이름 확인
cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(cat_feature_names)
print(f"\n전처리된 특성 이름: {all_feature_names}")

# DataFrame으로 확인
df_processed = pd.DataFrame(X_processed, columns=all_feature_names)
print(f"\n전처리 결과:")
print(df_processed.round(2))

'''
원본 shape: (8, 4)
전처리 후 shape: (8, 6)

전처리된 특성 이름: ['age', 'income', 'gender_F', 'gender_M', 'plan_basic', 'plan_premium']     

전처리 결과:
    age  income  gender_F  gender_M  plan_basic  plan_premium
0 -1.60   -1.54       0.0       1.0         1.0           0.0
1 -0.41   -0.07       1.0       0.0         0.0           1.0
2 -0.06   -0.37       0.0       1.0         1.0           0.0
3  1.95    1.97       1.0       0.0         1.0           0.0
4  0.77   -0.07       0.0       1.0         0.0           1.0
5 -0.89   -0.95       0.0       1.0         1.0           0.0
6 -0.06    0.80       1.0       0.0         0.0           1.0
7  0.30    0.22       0.0       1.0         1.0           0.0
'''


# ================================================================================
# Lv.1 기본 — 결측치 처리 연습

# [과제]
# 다음 데이터에서 결측치를 처리하라:

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

X = np.array([
    [25, 50000, 3],
    [np.nan, 60000, 5],
    [35, np.nan, np.nan],
    [40, 80000, 2],
    [30, 70000, 4],
])

df = pd.DataFrame(X)

# 1. 각 열의 결측치 수를 확인하라
print("=== 1. 열별 결측치 수 ===")
print(df.isnull().sum())
print("\n")

# 2. SimpleImputer(strategy='mean')으로 대체하라
imputer_mean = SimpleImputer(strategy='mean')
df_mean = pd.DataFrame(imputer_mean.fit_transform(df), columns=df.columns)

print("=== 2. 평균(Mean) 대체 결과 ===")
print(df_mean)
print("\n")

# 3. SimpleImputer(strategy='median')으로 대체하라
imputer_median = SimpleImputer(strategy='median')
df_median = pd.DataFrame(imputer_median.fit_transform(df), columns=df.columns)

print("=== 3. 중앙값(Median) 대체 결과 ===")
print(df_median)
print("\n")

# 4. 두 결과를 비교하고, 어떤 전략이 더 나은지 설명하라

'''
=== 1. 열별 결측치 수 ===
0    1
1    1
2    1
dtype: int64

=== 2. 평균(Mean) 대체 결과 ===
      0        1    2
0  25.0  50000.0  3.0
1  32.5  60000.0  5.0
2  35.0  65000.0  3.5
3  40.0  80000.0  2.0
4  30.0  70000.0  4.0

=== 3. 중앙값(Median) 대체 결과 ===
      0        1    2
0  25.0  50000.0  3.0
1  32.5  60000.0  5.0
2  35.0  65000.0  3.5
3  40.0  80000.0  2.0
4  30.0  70000.0  4.0
'''

# 극단적인 이상치가 존재하지 않음으로 어떤것을 이용하든 큰 문제가 없음.
# 단, 중앙값이 더 나은 전략임. 평균은 이상치에 크게 왜곡되지만, 중앙값은 중심 경향성을 잘 유지함.

# [힌트]
# - 이 데이터에서는 이상치가 없으므로 mean과 median 차이가 작음


# ================================================================================
# Lv.2 응용 — 타이타닉 데이터 전처리

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