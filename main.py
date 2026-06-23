import matplotlib.pyplot as plt
from matplotlib import rcParams

# -----------------------------------------------------------------
# 📌 한글 깨짐 해결 필수 설정 (사용하시는 OS에 맞는 폰트 하나만 선택하세요)
# -----------------------------------------------------------------

# 1. Windows 사용자의 경우
rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕 지정

# 2. macOS 사용자의 경우
# rcParams['font.family'] = 'AppleGothic'    # 애플고딕 지정

# 3. 리눅스(Ubuntu 등) / 구글 코랩(Colab) 사용자의 경우 (폰트가 설치되어 있어야 함)
# rcParams['font.family'] = 'NanumGothic'   # 나눔고딕 지정

# [필수] 마이너스 부호(-)가 네모나 깨진 글자로 출력되는 현상 방지
rcParams['axes.unicode_minus'] = False     
# -----------------------------------------------------------------

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