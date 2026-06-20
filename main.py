import matplotlib.pyplot as plt
from matplotlib import rcParams

# -----------------------------------------------------------------
# 📌 한글 깨짐 해결 필수 설정 (사용하시는 OS에 맞는 폰트 하나만 선택하세요)
# -----------------------------------------------------------------

# 1. Windows 사용자의 경우
# rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕 지정

# 2. macOS 사용자의 경우
rcParams['font.family'] = 'AppleGothic'    # 애플고딕 지정

# 3. 리눅스(Ubuntu 등) / 구글 코랩(Colab) 사용자의 경우 (폰트가 설치되어 있어야 함)
# rcParams['font.family'] = 'NanumGothic'   # 나눔고딕 지정

# [필수] 마이너스 부호(-)가 네모나 깨진 글자로 출력되는 현상 방지
rcParams['axes.unicode_minus'] = False     
# -----------------------------------------------------------------


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

