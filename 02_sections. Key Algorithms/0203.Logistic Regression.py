# 코드 예시
# 2-1. 기본 로지스틱 회귀

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
log_reg = LogisticRegression(C=1.0, max_iter=200, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = log_reg.predict(X_test_scaled)
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=cancer.target_names)}")

'''
정확도: 0.9825

              precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
'''


# ================================================================================
# 2-2. 확률 예측 (predict_proba)



# ================================================================================
# 2-3. 계수(가중치) 해석




# ================================================================================
# 2-4.  규제 강도 C 비교




# ================================================================================
# 2-5. 다중 클래스 분류




# ================================================================================
# Lv.1 기본 — 결측치 처리 연습




# ================================================================================
# Lv.2 응용 — 타이타닉 데이터 전처리


