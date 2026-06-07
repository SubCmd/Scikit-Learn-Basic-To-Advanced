# 코드 예시
# 2-1. 기본 결정 트리 + 시각화

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

# 모델 학습 (스케일링 불필요!)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")

# 텍스트 기반 트리 규칙 출력
print("\n=== 결정 규칙 ===")
print(export_text(dt, feature_names=list(iris.feature_names)))

# 시각화
plt.figure(figsize=(16, 8))
plot_tree(dt, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, rounded=True,
          fontsize=10)
plt.title("Decision Tree - Iris Dataset")
plt.tight_layout()
plt.savefig("00_download/decision_tree_iris.png", dpi=150)
plt.close()

'''
정확도: 0.9667

=== 결정 규칙 ===
|--- petal length (cm) <= 2.45
|   |--- class: 0
|--- petal length (cm) >  2.45
|   |--- petal width (cm) <= 1.65
|   |   |--- petal length (cm) <= 4.95
|   |   |   |--- class: 1
|   |   |--- petal length (cm) >  4.95
|   |   |   |--- class: 2
|   |--- petal width (cm) >  1.65
|   |   |--- petal length (cm) <= 4.85
|   |   |   |--- class: 2
|   |   |--- petal length (cm) >  4.85
|   |   |   |--- class: 2
'''


# ================================================================================
# 2-2. 특성 중요도 분석

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

# 모델 학습
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# 특성 중요도
importance_df = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("=== 특성 중요도 ===")
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:25s} | {row['importance']:.4f} | {bar}")

# 시각화
plt.figure(figsize=(8, 4))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Decision Tree')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("00_download/feature_importance_dt.png", dpi=150)
plt.close()

'''
=== 특성 중요도 ===
  petal length (cm)         | 0.5791 | ████████████████████████████
  petal width (cm)          | 0.4209 | █████████████████████
  sepal length (cm)         | 0.0000 | 
  sepal width (cm)          | 0.0000 | 
'''

# ================================================================================
# 2-3. 과적합 실험: max_depth 변화

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

print(f"{'max_depth':>10} | {'훈련 정확도':>10} | {'테스트 정확도':>10} | {'갭':>10} ")
print("-" * 50)

for depth in [1, 2, 3, 5, 10, 20, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)
    gap = train_acc - test_acc
    depth_str = str(depth) if depth else 'None'
    print(f"{depth_str:>10} | {train_acc:>10.4f} | {test_acc:>10.4f} | {gap:>8.4f}")

# 예상 패턴:
# max_depth=None → 훈련 100%, 테스트 낮음 (과적합)
# max_depth=3~5  → 훈련과 테스트 갭이 적음 (적절)
# max_depth=1    → 둘 다 낮음 (과소적합)

'''
 max_depth |     훈련 정확도 |    테스트 정확도 |          갭
--------------------------------------------------
         1 |     0.9231 |     0.9211 |   0.0020
         2 |     0.9582 |     0.8947 |   0.0635
         3 |     0.9758 |     0.9386 |   0.0372
         5 |     0.9934 |     0.9211 |   0.0724
        10 |     1.0000 |     0.9123 |   0.0877
        20 |     1.0000 |     0.9123 |   0.0877
      None |     1.0000 |     0.9123 |   0.0877
'''


# ================================================================================
# 2-4.  결정 트리 회귀

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# max_depth별 비교
for depth in [3, 5, 10, None]:
    dt_reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt_reg.fit(X_train, y_train)
    y_pred = dt_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    depth_str = str(depth) if depth else 'None'
    print(f"max_depth={depth_str:>5} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

'''
max_depth=    3 | RMSE: 0.8015 | R²: 0.5098
max_depth=    5 | RMSE: 0.7242 | R²: 0.5997
max_depth=   10 | RMSE: 0.6446 | R²: 0.6829
max_depth= None | RMSE: 0.7069 | R²: 0.6187
'''


# ================================================================================
# Lv.1 기본 — 결정 트리 적용 및 규칙 추출

# [과제]
# load_wine() 데이터에 대해:

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, 
)

# 1. DecisionTreeClassifier(max_depth=3)를 학습시켜라
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# 2. 정확도와 classification_report를 출력하라
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 데이터 정확도: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 3. export_text()로 트리 규칙을 텍스트로 출력하라
print(export_text(dt, feature_names=list(wine.feature_names)))

# 4. 특성 중요도 상위 5개를 출력하라
importance_df = pd.DataFrame({
    'feature': wine.feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(5))

# 5. 중요도가 0인 특성이 있는지 확인하라 (트리가 사용하지 않은 특성)
unused_features = importance_df[importance_df['importance'] == 0]
print(f"중요도가 0인 특성 개수: {len(unused_features)}개")
print(unused_features['feature'].to_list())

# [힌트]
# - max_depth=3이면 일부 특성은 사용되지 않을 수 있음
# - feature_importances_에서 0인 값 확인

'''
테스트 데이터 정확도: 0.8667
              precision    recall  f1-score   support

     class_0       0.94      1.00      0.97        16
     class_1       0.90      0.86      0.88        21
     class_2       0.62      0.62      0.62         8

    accuracy                           0.87        45
   macro avg       0.82      0.83      0.82        45
weighted avg       0.87      0.87      0.87        45

|--- od280/od315_of_diluted_wines <= 2.19
|   |--- malic_acid <= 1.64
|   |   |--- magnesium <= 116.00
|   |   |   |--- class: 1
|   |   |--- magnesium >  116.00
|   |   |   |--- class: 2
|   |--- malic_acid >  1.64
|   |   |--- class: 2
|--- od280/od315_of_diluted_wines >  2.19
|   |--- proline <= 726.50
|   |   |--- flavanoids <= 0.75
|   |   |   |--- class: 2
|   |   |--- flavanoids >  0.75
|   |   |   |--- class: 1
|   |--- proline >  726.50
|   |   |--- color_intensity <= 3.43
|   |   |   |--- class: 1
|   |   |--- color_intensity >  3.43
|   |   |   |--- class: 0

                         feature  importance
12                       proline    0.443093
11  od280/od315_of_diluted_wines    0.406435
1                     malic_acid    0.065525
9                color_intensity    0.044235
6                     flavanoids    0.022175
중요도가 0인 특성 개수: 7개
['alcohol', 'ash', 'alcalinity_of_ash', 'total_phenols', 'nonflavanoid_phenols', 'proanthocyanins', 'hue']
'''


# ================================================================================
# Lv.2 응용 — 가지치기 파라미터 튜닝


