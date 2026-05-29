# 코드 예시
# 2-1. KNN 분류 (기본)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

# 스케일링 (KNN은 거리 기반이므로 필수!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit & transform 동시에 이용 
X_test_scaled = scaler.transform(X_test)        # transform만 이용 : fit이 되면 모델 학습 과정에서 data leakage 발생
## 즉, test 데이터에서 fit하면 미리 예측하는 문제가 발생해 (일반화 성능)을 왜곡하게 됨.

# 모델 학습 및 예측
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# 평가
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")

'''
정확도: 0.9333

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.83      1.00      0.91        10
   virginica       1.00      0.80      0.89        10

    accuracy                           0.93        30
   macro avg       0.94      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30
'''


# ================================================================================
# 2-2. 최적 K 탐색

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터 준비
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# K=1~30까지 탐색
k_range = range(1, 31)
train_scores = []
test_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    train_scores.append(knn.score(X_train_s, y_train))
    test_scores.append(knn.score(X_test_s, y_test))

# 최적 K 찾기
best_k = k_range[test_scores.index(max(test_scores))]
print(f"최적 K: {best_k}, 테스트 정확도: {max(test_scores):.4f}")

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(k_range, train_scores, 'o-', label='Train Accuracy')
plt.plot(k_range, test_scores, 's-', label='Test Accuracy')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('K (n_neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: K값에 따른 정확도 변화')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("00_download\knn_k_search.png", dpi=150)
plt.close()

'''
최적 K: 3, 테스트 정확도: 0.9825
'''


# ================================================================================
# 2-3. KNN 회귀

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 데이터 준비
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42,
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
## 데이터의 형태를 맞추기 위한 학습 `fit` 

# KNN 회귀
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train_s, y_train)
y_pred = knn_reg.predict(X_test_s)
## 정답을 맞추기 위한 학습 `fit`

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

'''
RMSE: 0.6557
R²:   0.6719
'''


# ================================================================================
# 2-4.  weights='uniform' vs 'distance' 비교

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)
# stratify=y : 가독성을 위해 사용
# stratify=wine.target : 에러 방지를 위한 명시화 

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for weight in ['uniform', 'distance']:
    knn = KNeighborsClassifier(n_neighbors=5, weights=weight)
    scores = cross_val_score(knn, X_train_s, y_train, cv=5, scoring='accuracy')
    knn.fit(X_train_s, y_train)
    test_acc = knn.score(X_test_s, y_test)
    print(f"weights='{weight}' | CV: {scores.mean():.4f}±{scores.std():.4f} | Test: {test_acc:.4f}")

'''
weights='uniform' | CV: 0.9510±0.0354 | Test: 0.9722
weights='distance' | CV: 0.9510±0.0354 | Test: 0.9722
'''


# ================================================================================
# Lv.1 기본 — KNN 적용 및 평가

# [과제]
# load_wine() 데이터셋에 KNN 분류를 적용하라:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. 데이터를 80:20으로 분리하고 StandardScaler를 적용하라
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 2. K=3으로 KNN 모델을 학습시키고 정확도를 출력하라
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

# 3. classification_report를 출력하라
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=wine.target_names)}")

# 4. predict_proba()로 테스트 첫 5개 샘플의 클래스별 확률을 출력하라
proba = knn.predict_proba(X_test_s[:5])
proba_df = pd.DataFrame(proba, columns=wine.target_names)
proba_df.index = [f"샘플 {i+1} (실제: {wine.target_names[y_test[i]]})" for i in range(5)]

print("=== 테스트 첫 5개 샘플의 클래스별 확률 ===")
print(proba_df)


'''
정확도: 0.9722

              precision    recall  f1-score   support

     class_0       0.92      1.00      0.96        12
     class_1       1.00      0.93      0.96        14
     class_2       1.00      1.00      1.00        10

    accuracy                           0.97        36
   macro avg       0.97      0.98      0.97        36
weighted avg       0.97      0.97      0.97        36

=== 테스트 첫 5개 샘플의 클래스별 확률 ===
                     class_0   class_1   class_2
샘플 1 (실제: class_0)  1.000000  0.000000  0.000000
샘플 2 (실제: class_2)  0.000000  0.333333  0.666667
샘플 3 (실제: class_0)  1.000000  0.000000  0.000000
샘플 4 (실제: class_1)  0.666667  0.333333  0.000000
샘플 5 (실제: class_1)  0.000000  1.000000  0.000000
'''


# ================================================================================
# Lv.2 응용 — 최적 K 탐색 + 교차 검증

