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