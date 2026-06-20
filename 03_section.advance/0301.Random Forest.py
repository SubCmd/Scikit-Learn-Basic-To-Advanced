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
# 2-2. Ridge vs. Lasso vs. ElasticNet 비교





# ================================================================================
# 2-3. Lasso의 특성 선택 효과




# ================================================================================
# 2-4.  alpha 최적값 탐색 (교차 검증)





# ================================================================================
# 2-5.  회귀 모델 종합 비교





# ================================================================================
# Lv.1 기본 - 선형 회귀 적용 및 해석




