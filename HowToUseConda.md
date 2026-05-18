## 1단계. 기본 설정 및 업데이트

conda update -n base -c defaults conda
conda update --all


## 2단계. 가상환경 생성
- 새 환경 생성 (python 버전 지정)X
conda create -n myenv python=3.11

- 환경 목록 확인
conda env list

- 환경 활성화
conda activate myenv

- 환경 비활성화
conda deactivate

- 환경 삭제
conda remove -n myenv --all


## 3단계. 패키지
- conda로 설치 (권장)
conda install numpy pandas matplotlib scikit-learn

- 특정 채널에서 설치
conda install -c conda-forge jupyterlab

- pip 사용 (conda에 없는 경우만)
pip install <패키지명>

- 설치된 패키지 확인
conda list


## 4단계. 환경 내보내기 / 공유
- 환경 export (재현용)
conda env export > environment.yml

- 환경 파일로부터 재생성
conda env create -f environment.yml