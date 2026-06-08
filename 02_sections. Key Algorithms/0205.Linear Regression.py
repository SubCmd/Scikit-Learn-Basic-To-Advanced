# 코드 예시
# 2-1. 기본 선형 회귀

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 데이터 준비
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# 선형 회귀 학습
lr = LinearRegression()
lr.fit(X_train_scaler, y_train)
y_pred = lr.predict(X_test_scaler)

# 평가
print("=== 선형 회귀 성능 ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred):4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# 계수 해석
coef_df = pd.DataFrame({
    'feature': housing.feature_names,
    'coefficient': lr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\n절편(intercept): {lr.intercept_:.4f}")
print(f"\n=== 회귀 계수 (절대값 내림차순) ===")
for _, row in coef_df.iterrows():
    direction = "↑" if row['coefficient'] > 0 else "↓"
    print(f"  {row['feature']:15s} | {row['coefficient']:+.4f} | 가격 {direction}")

'''
=== 선형 회귀 성능 ===
MAE:  0.5332
RMSE: 0.555892
R²:   0.5758

절편(intercept): 2.0719

=== 회귀 계수 (절대값 내림차순) ===
  Latitude        | -0.8969 | 가격 ↓
  Longitude       | -0.8698 | 가격 ↓
  MedInc          | +0.8544 | 가격 ↑
  AveBedrms       | +0.3393 | 가격 ↑
  AveRooms        | -0.2944 | 가격 ↓
  HouseAge        | +0.1225 | 가격 ↑
  AveOccup        | -0.0408 | 가격 ↓
  Population      | -0.0023 | 가격 ↓
'''


# ================================================================================
# 2-2. Ridge vs. Lasso vs. ElasticNet 비교

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 데이터 준비
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge(α=1.0)': Ridge(alpha=1.0),
    'Lasso(α=0.1)': Lasso(alpha=0.1),
    'ElasticNet(α=0.1)': ElasticNet(alpha=0.1, l1_ratio=0.5),
}

print(f"{'모델':25s} | {'RMSE':>8} | {'R²':>8} | {'비제로 계수':>10}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train_scaler, y_train)
    y_pred = model.predict(X_test_scaler)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
    print(f"{name:25s} | {rmse:>8.4f} | {r2:>8.4f} | {n_nonzero:>10}/{len(model.coef_)}")

'''
모델                        |     RMSE |       R² |     비제로 계수
-----------------------------------------------------------------
LinearRegression          |   0.7456 |   0.5758 |          8/8
Ridge(α=1.0)              |   0.7456 |   0.5758 |          8/8
Lasso(α=0.1)              |   0.8244 |   0.4814 |          3/8
ElasticNet(α=0.1)         |   0.7974 |   0.5148 |          4/8
'''


# ================================================================================
# 2-3. Lasso의 특성 선택 효과

from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 데이터 준비
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# alpha 값에 따른 특성 선택 변화
alphas = [0.001, 0.01, 0.1, 0.5, 1.0]

print("=== Lasso: alpha별 비제로 계수 수 ===")
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaler, y_train)
    n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-10)
    r2 = r2_score(y_test, lasso.predict(X_test_scaler))
    print(f"  alpha={alpha:>5} | 비제로 계수: {n_nonzero}/{len(lasso.coef_)} | R²: {r2:.4f}")

# 상세 계수 확인 (alpha=0.1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaler, y_train)

coef_df = pd.DataFrame({
    'feature': housing.feature_names,
    'coefficient': lasso.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\n=== Lasso(α=0.1) 계수 ===")
for _, row in coef_df.iterrows():
    status = "✅ 유지" if abs(row['coefficient']) > 1e-10 else "❌ 제거"
    print(f"  {row['feature']:15s} | {row['coefficient']:+.4f} | {status}")

'''
=== Lasso: alpha별 비제로 계수 수 ===
  alpha=0.001 | 비제로 계수: 8/8 | R²: 0.5769
  alpha= 0.01 | 비제로 계수: 7/8 | R²: 0.5816
  alpha=  0.1 | 비제로 계수: 3/8 | R²: 0.4814
  alpha=  0.5 | 비제로 계수: 1/8 | R²: 0.2827
  alpha=  1.0 | 비제로 계수: 0/8 | R²: -0.0002

=== Lasso(α=0.1) 계수 ===
  MedInc          | +0.7106 | ✅ 유지
  HouseAge        | +0.1065 | ✅ 유지
  Latitude        | -0.0115 | ✅ 유지
  AveRooms        | -0.0000 | ❌ 제거
  AveBedrms       | +0.0000 | ❌ 제거
  Population      | -0.0000 | ❌ 제거
  AveOccup        | -0.0000 | ❌ 제거
  Longitude       | -0.0000 | ❌ 제거
'''


# ================================================================================
# 2-4.  alpha 최적값 탐색 (교차 검증)

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 데이터 준비
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# === RidgeCV: 내장 교차 검증으로 최적 alpha 자동 탐색 ===
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_cv.fit(X_train_scaler, y_train)
print(f"Ridge 최적 alpha: {ridge_cv.alpha_}")
print(f"Ridge R²: {ridge_cv.score(X_test_scaler, y_test):.4f}")

# === LassoCV: 내장 교차 검증 ===
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 0.5, 1.0], cv=5)
lasso_cv.fit(X_train_scaler, y_train)
print(f"\nLasso 최적 alpha: {lasso_cv.alpha_}")
print(f"Lasso R²: {lasso_cv.score(X_test_scaler, y_test):.4f}")
n_selected = np.sum(np.abs(lasso_cv.coef_) > 1e-10)
print(f"Lasso 선택된 특성 수: {n_selected}/{len(lasso_cv.coef_)}")

'''
Ridge 최적 alpha: 0.01
Ridge R²: 0.5758

Lasso 최적 alpha: 0.001
Lasso R²: 0.5769
Lasso 선택된 특성 수: 8/8
'''


# ================================================================================
# 2-5.  회귀 모델 종합 비교

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 데이터 준비
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge(α=1.0)': Ridge(alpha=1.0),
    'Lasso(α=0.01)': Lasso(alpha=0.01),
    'DecisionTree(d=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
    'KNN(K=5)': KNeighborsRegressor(n_neighbors=5),
}

print(f"{'모델':25s} | {'CV R² 평균':>10} | {'CV R² 표준편차':>12}")
print("-" * 55)

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaler, y_train, cv=5, scoring='r2')
    print(f"{name:25s} | {scores.mean():>10.4f} | {scores.std():>12.4f}")

'''
모델                        |   CV R² 평균 |   CV R² 표준편차
-------------------------------------------------------
LinearRegression          |     0.6115 |       0.0065
Ridge(α=1.0)              |     0.6115 |       0.0065
Lasso(α=0.01)             |     0.6077 |       0.0047
DecisionTree(d=5)         |     0.6198 |       0.0136
KNN(K=5)                  |     0.6836 |       0.0066
'''


# ================================================================================
# Lv.1 기본 - 선형 회귀 적용 및 해석

# [과제]
# fetch_california_housing() 데이터에 대해:

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 데이터 준비
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# 1. LinearRegression을 학습시키고 MAE, RMSE, R²를 출력하라
lr = LinearRegression()
lr.fit(X_train_scaler, y_train)
y_pred = lr.predict(X_test_scaler)

print("=== 선형 회귀 성능 ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred):4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# 2. 각 특성의 회귀 계수를 절대값 내림차순으로 출력하라
features = housing.feature_names
coefficients = lr.coef_

df_coef = pd.DataFrame({
    'Feature': features,
    'Coef': coefficients,
    'Abs_Coef': np.abs(coefficients)
})

# 절대값 내림차순 정렬
df_coef_sorted = df_coef.sort_values(by='Abs_Coef', ascending=False)

print("\n=== 2. 회귀 계수 (절대값 내림차순) ===")
for idx, row in df_coef_sorted.iterrows():
    print(f"{row['Feature']:<12} : 계수 = {row['Coef']:8.4f} (절대값 = {row['Abs_Coef']:.4f})")

# 3. 양의 계수(가격↑)와 음의 계수(가격↓)를 분류하라

print("\n=== 3. 양의 계수와 음의 계수 분류 ===")
pos_features = df_coef[df_coef['Coef'] > 0]
neg_features = df_coef[df_coef['Coef'] < 0]

print("[양의 계수 (가격 ↑)]")
for idx, row in pos_features.sort_values(by='Coef', ascending=False).iterrows():
    print(f" - {row['Feature']}: {row['Coef']:.4f}")

print("\n[음의 계수 (가격 ↓)]")
for idx, row in neg_features.sort_values(by='Coef', ascending=True).iterrows():
    print(f" - {row['Feature']}: {row['Coef']:.4f}")

# 4. "MedInc(중위소득)가 1 표준편차 증가하면 주택 가격이
#    약 X만 달러 변화한다"는 형식으로 상위 3개 특성을 해석하라

print("\n=== 4. 상위 3개 특성 해석 ===")
# 캘리포니아 주택 데이터의 타깃(y) 단위는 '10만 달러'입니다.
# 따라서 'X만 달러' 형식으로 맞추기 위해 계수에 10을 곱해줍니다. (1단위 = 10만 달러 = 100,000달러)
top_3 = df_coef_sorted.head(3)

for idx, row in top_3.iterrows():
    feature_name = row['Feature']
    coef_val = row['Coef']
    
    # 양수/음수에 따른 표현 처리
    direction = "증가" if coef_val > 0 else "감소"
    # 10만 달러 단위에서 '만 달러' 단위로 변환 (절대값 적용)
    change_in_ten_thousand = abs(coef_val) * 10 
    
    print(f" - \"{feature_name}가 1 표준편차 증가하면 주택 가격이 약 {change_in_ten_thousand:.1f}만 달러 {direction}한다\"")

'''
=== 선형 회귀 성능 ===
MAE:  0.5332
RMSE: 0.555892
R²:   0.5758

=== 2. 회귀 계수 (절대값 내림차순) ===
Latitude     : 계수 =  -0.8969 (절대값 = 0.8969)
Longitude    : 계수 =  -0.8698 (절대값 = 0.8698)
MedInc       : 계수 =   0.8544 (절대값 = 0.8544)
AveBedrms    : 계수 =   0.3393 (절대값 = 0.3393)
AveRooms     : 계수 =  -0.2944 (절대값 = 0.2944)
HouseAge     : 계수 =   0.1225 (절대값 = 0.1225)
AveOccup     : 계수 =  -0.0408 (절대값 = 0.0408)
Population   : 계수 =  -0.0023 (절대값 = 0.0023)

=== 3. 양의 계수와 음의 계수 분류 ===
[양의 계수 (가격 ↑)]
 - MedInc: 0.8544
 - AveBedrms: 0.3393
 - HouseAge: 0.1225

[음의 계수 (가격 ↓)]
 - Latitude: -0.8969
 - Longitude: -0.8698
 - AveRooms: -0.2944
 - AveOccup: -0.0408
 - Population: -0.0023

=== 4. 상위 3개 특성 해석 ===
 - "Latitude가 1 표준편차 증가하면 주택 가격이 약 9.0만 달러 감소한다"
 - "Longitude가 1 표준편차 증가하면 주택 가격이 약 8.7만 달러 감소한다"
 - "MedInc가 1 표준편차 증가하면 주택 가격이 약 8.5만 달러 증가한다"
'''