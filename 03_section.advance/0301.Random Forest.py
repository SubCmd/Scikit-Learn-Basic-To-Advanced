# 코드 예시
# 2-1. 기본 랜덤 포레스트 분류

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

# 스케일링 불필요!
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    oob_score=True       # OOB 평가 활성화
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"테스트 정확도: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB 점수:     {rf.oob_score_:.4f}")  # 무료 검증 점수
print(f"\n{classification_report(y_test, y_pred, target_names=cancer.target_names)}")

'''
테스트 정확도: 0.9561
OOB 점수:     0.9538

              precision    recall  f1-score   support

   malignant       0.95      0.93      0.94        42
      benign       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114
'''


# ================================================================================
# 2-2. 특성 중요도 (안정적)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

# 스케일링 불필요!
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features='sqrt',
    random_state=42,
    n_jobs=1,
    oob_score=True  # OOB 평가 활성화
)
rf.fit(X_train, y_train)

# 특성 중요도 추출
importance_df = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Top 10 출력
print("=== 특성 중요도 Top 10 ===")
for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
    bar = '█' * int(row['importance'] * 100)
    print(f"  {i+1:2d}. {row['feature']:30s} | {row['importance']:.4f} | {bar}")

# 시각화
plt.figure(figsize=(10, 6))
top10 = importance_df.head(10)
plt.barh(range(len(top10)), top10['importance'])
plt.yticks(range(len(top10)), top10['feature'])
plt.xlabel('Importance')
plt.title('Random Forest - Feature Importance Top 10')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150)
plt.close()

'''
=== 특성 중요도 Top 10 ===
   1. worst area                     | 0.1400 | ██████████████
   2. worst concave points           | 0.1295 | ████████████
   3. worst radius                   | 0.0977 | █████████
   4. mean concave points            | 0.0909 | █████████
   5. worst perimeter                | 0.0722 | ███████
   6. mean perimeter                 | 0.0696 | ██████
   7. mean radius                    | 0.0687 | ██████
   8. mean concavity                 | 0.0576 | █████
   9. mean area                      | 0.0492 | ████
  10. worst concavity                | 0.0343 | ███
'''


# ================================================================================
# 2-3. n_estimators(트리 수)에 따른 성능 변화

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

# 스케일링 불필요!
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features='sqrt',
    random_state=42,
    n_jobs=1,
    oob_score=True  # OOB 평가 활성화
)
rf.fit(X_train, y_train)

n_trees_list = [1, 5, 10, 30, 50, 100, 200, 300, 500]
train_accs = []
test_accs = []

for n_trees in n_trees_list:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_accs.append(rf.score(X_train, y_train))
    test_accs.append(rf.score(X_test, y_test))
    print(f"n_estimators={n_trees:>3d} | Train: {train_accs[-1]:.4f} | Test: {test_accs[-1]:.4f}")

# 트리 수 증가에 따라 테스트 정확도가 안정화됨 (수렴)
plt.figure(figsize=(8, 5))
plt.plot(n_trees_list, train_accs, 'o-', label='Train')
plt.plot(n_trees_list, test_accs, 's-', label='Test')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest: 트리 수에 따른 정확도')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("00_download/rf_n_estimators.png", dpi=150)
plt.close()

'''
n_estimators=  1 | Train: 0.9714 | Test: 0.9474
n_estimators=  5 | Train: 0.9912 | Test: 0.9386
n_estimators= 10 | Train: 0.9956 | Test: 0.9386
n_estimators= 30 | Train: 1.0000 | Test: 0.9474
n_estimators= 50 | Train: 1.0000 | Test: 0.9561
n_estimators=100 | Train: 1.0000 | Test: 0.9561
n_estimators=200 | Train: 1.0000 | Test: 0.9561
n_estimators=300 | Train: 1.0000 | Test: 0.9474
n_estimators=500 | Train: 1.0000 | Test: 0.9561
'''


# ================================================================================
# 2-4.  랜덤포레스트 회귀

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")


'''
RMSE: 0.5053
R²:   0.8051
'''


# ================================================================================
# 2-5.  결정 트리 vs. 랜덤 포레스트 안정성 비교

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)


# 여러 random_state로 분산 비교
dt_scores = []
rf_scores = []

for seed in range(20):
    dt = DecisionTreeClassifier(max_depth=5, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1)

    dt_cv = cross_val_score(dt, X_train, y_train, cv=5).mean()
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5).mean()

    dt_scores.append(dt_cv)
    rf_scores.append(rf_cv)

import numpy as np
print(f"Decision Tree | 평균: {np.mean(dt_scores):.4f} ± {np.std(dt_scores):.4f}")
print(f"Random Forest | 평균: {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
# → 랜덤 포레스트가 더 높은 평균, 더 낮은 분산 (안정적)

'''
Decision Tree | 평균: 0.9243 ± 0.0040
Random Forest | 평균: 0.9547 ± 0.0037
'''



# ================================================================================
# Lv.1 기본 - 선형 회귀 적용 및 해석

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# [과제]
# load_wine() 데이터에 대해:

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)

# 1. RandomForestClassifier(n_estimators=100, random_state=42)를 학습하라

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
rf.fit(X_train, y_train)

# 2. 정확도와 OOB 점수를 출력하라
y_pred_rf = rf.predict(X_test)
print(f"테스트 정확도: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"OOB 점수:     {rf.oob_score_:.4f}")  # 무료 검증 점수
print(f"\n{classification_report(y_test, y_pred_rf, target_names=wine.target_names)}")

# 3. 특성 중요도 상위 5개를 출력하라
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(5):
    idx = indices[i]
    print(f"{i+1}위: {wine.feature_names[idx]} ({importances[idx]:.4f})")
print("\n" + "-" * 50 + "\n")

# 4. DecisionTreeClassifier(max_depth=5)와 정확도를 비교하라
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print(f"RandomForest 정확도: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"DecisionTree 정확도: {accuracy_score(y_test, y_pred_dt):.4f}\n")

# 5. 5-Fold 교차 검증으로 두 모델의 평균±표준편차를 비교하라

cv_scores_rf = cross_val_score(
    rf, wine.data, wine.target, cv=5, scoring="accuracy"
)
cv_scores_dt = cross_val_score(
    dt, wine.data, wine.target, cv=5, scoring="accuracy"
)

print(
    f"RandomForest CV 정확도: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}"
)
print(
    f"DecisionTree CV 정확도: {cv_scores_dt.mean():.4f} ± {cv_scores_dt.std():.4f}"
)

# [힌트]
# - oob_score=True 설정
# - 랜덤 포레스트가 더 높은 CV 점수와 낮은 표준편차를 보일 것

'''
테스트 정확도: 1.0000
OOB 점수:     0.9789

              precision    recall  f1-score   support

     class_0       1.00      1.00      1.00        12
     class_1       1.00      1.00      1.00        14
     class_2       1.00      1.00      1.00        10

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36

1위: color_intensity (0.1876)
2위: flavanoids (0.1596)
3위: proline (0.1468)
4위: alcohol (0.1179)
5위: hue (0.1015)

--------------------------------------------------

RandomForest 정확도: 1.0000
DecisionTree 정확도: 0.9444

RandomForest CV 정확도: 0.9721 ± 0.0176
DecisionTree CV 정확도: 0.8654 ± 0.0440
'''


