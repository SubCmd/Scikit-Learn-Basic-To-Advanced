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




# ================================================================================
# 2-4.  랜덤포레스트 회귀





# ================================================================================
# 2-5.  결정 트리 vs. 랜덤 포레스트 안정성 비교





# ================================================================================
# Lv.1 기본 - 선형 회귀 적용 및 해석




