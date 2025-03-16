# 🚀 **storagy-repo-5**

## 📌 **1️⃣ 프로젝트 개요**

`storagy-repo-5`는 **건물 내 배달 로봇 및 스토리지 로봇**이 특정 용도에 국한되지 않고, 
**필수 기능과 사용자 편의성에 초점을 맞춘 서비스**를 제공하도록 개발된 서비스 패키지의 리포지토리입니다.

**이 프로젝트는 다음과 같은 주요 기능을 포함합니다:**

- 🤖 **로봇 호출**: 사용자가 특정 위치에서 로봇을 호출하고 목적지까지 이동할 수 있도록 지원
- 👤 **사용자 트래킹**: 로봇이 실시간으로 사용자의 위치를 추적하여 효율적인 경로를 결정
- 🏢 **평면도 맵 변환**: 업로드된 건물 도면을 로봇 네비게이션을 위한 **PGM 맵**으로 자동 변환
- 🗺 **자동 매핑**: 기존 맵이 없는 공간에서도 **로봇이 직접 매핑**을 수행하여 경로를 탐색

위 기능은 다음과 같은 기술을 반영합니다:

- 📡 웹 서버 기반: 메인 스크립트 실행 시 **Flask** 서버 자동 시작
- 📂 AI 기반 도면 전처리: **Pix2Pix**로 전처리 후, 해당 이미지로 YAML 파일 및 PGM 파일 생성
- 🚀 대용량 모델 관리: **HuggingFace** 저장소와 패키지 이용
- 🔍 객체 인식: **SAM2**를 이용하여 영상 내 특정 객체 감지

---

## 📌 **2️⃣ 프로젝트 구조**

```bash
📦 storagy-repo-5
┣ 📂 ai
┃ ┣ 📂 checkpoint_a2b_inorm
┃ ┃ ┗ 🗂 model_epoch400.pth → 사전 학습된 모델 가중치
┃ ┣ 📂 log_a2b_inorm
┃ ┃ ┣ 📂 train → 학습 로그 데이터
┃ ┃ ┗ 📂 val → 검증 로그 데이터
┃ ┣ 📜 dataset.py → 데이터셋 구성 및 전처리
┃ ┣ 📜 layer.py → 신경망 레이어 정의
┃ ┣ 📜 model.py → 모델 구조 정의
┃ ┣ 📜 train.py → 모델 학습 및 검증
┃ ┣ 📜 util.py → 유틸리티 함수 모음
┃ ┗ 📜 main.py → 모델 실행 및 제어
┣ 📂 SAM2_streaming
┃ ┣ 📂 checkpoints
┃ ┃ ┗ 📂 sam2
┃ ┃   ┗ 🗂 sam2_hiera_tiny.pth → SAM2 모델 가중치
┃ ┣ 📂 configs
┃ ┃ ┗ 📂 sam2
┃ ┃   ┗ 📝 sam2_hiera_t.yaml → 모델 설정 파일
┃ ┣ 📂 sam2
┃ ┃ ┣ 📂 modeling → 모델 구현 모듈
┃ ┃ ┣ 📜 build_sam.py → SAM2 모델 구축 로직
┃ ┃ ┣ 📜 sam2_camera_predictor.py → 카메라 스트림 객체 추적
┃ ┃ ┗ 📜 sam2_image_predictor.py → 이미지 기반 객체 세그멘테이션
┃ ┗ 📜 demo_webcam_point.py → 웹캠 데모 애플리케이션
┣ 📂 pages
┃ ┣ 📜 guideline.py → 가이드라인 페이지 기능
┃ ┣ 📜 nav.py → 네비게이션 페이지 기능
┃ ┣ 📜 remap.py → 맵 수정 페이지 기능
┃ ┗ 📜 tracking.py → 객체 추적 페이지 기능
┣ 📂 robot
┃ ┣ 📜 path_planning.py → 경로 계획 알고리즘
┃ ┗ 📜 robot.py → 로봇 제어 인터페이스 (스토리지 로봇 패키지에 삽입)
┣ 📂 static
┃ ┣ 📂 css
┃ ┃ ┗ 🎨 styles.css → 스타일시트 파일
┃ ┣ 📂 images
┃ ┃ ┗  📷 social_share.jpg → 페이지 배경화면 이미지
┃ ┣ 📂 js
┃ ┃ ┗  📜 audio.js → 오디오 자바스크립트 파일
┃ ┣ 📂 map
┃ ┃ ┣ 🗺 map.pgm → 로봇 네비게이션용 맵 이미지
┃ ┃ ┣ 📷 map.png → 웹 표시용 맵 이미지
┃ ┃ ┗ 📝 map.yaml → 맵 메타데이터
┃ ┣ 📂 sounds
┃ ┃ ┣ 🎵 click1.mp3 → 웹 페이지 효과음
┃ ┃ ┗ 🎵 hover1.mp3 → 웹 페이지 효과음
┣ 📂 templates
┃ ┣ 🌍 guideline.html → 가이드라인 페이지 템플릿
┃ ┣ 🌍 main_server.html → 메인 페이지 템플릿
┃ ┣ 🌍 nav.html → 네비게이션 페이지 템플릿
┃ ┣ 🌍 remap.html → 맵 수정 페이지 템플릿
┃ ┗ 🌍 tracking.html → 객체 추적 페이지 템플릿
┣ 📂 utils
┃ ┣ 📜 config.py → 시스템 전역 설정
┃ ┗ 📜 download_large_files.py → 대용량 모델 파일 다운로드 유틸리티
┣ 📜 main_server.py → 웹 서버 메인 실행 파일
┣ 📜 requirements.txt → 필요 패키지 목록
┗ 📜 README.md → 프로젝트 설명 및 문서
```

---

## 📌 **3️⃣ 테스트 환경**

아래 환경에서 프로젝트가 테스트되었습니다.

| 항목            | 세부 정보 |
|----------------|----------|
| **OS**        | Ubuntu 22.04.5 LTS x86_64 |
| **CPU**       | AMD Ryzen 7 7735HS with Radeon Graphics (16) @ 4.829GHz |
| **GPU 1**     | AMD ATI 05:00.0 Rembrandt |
| **GPU 2**     | NVIDIA 01:00.0 NVIDIA Corporation Device 28a0 |
| **RAM Memory** | 15186MiB |
| **CUDA Version** | 12.6 |
| **GPU Memory** | 8188MiB |
| **NVIDIA-SMI Driver Version** | 560.35.05 |
| **Python Version** | 3.10.12 |
| **필수 패키지** | `requirements.txt` |

---

## 📌 **4️⃣ 설치 및 실행 방법**

### **💡 기본 설치 및 실행**
1️⃣ **프로젝트 클론 및 디렉토리 이동**
```bash
git clone https://github.com/addinedu-advance-3rd/storagy-repo-5.git
cd storagy-repo-5
```
2️⃣ **필수 패키지 설치**
```bash
pip install -r requirements.txt
```
✅ 이제 모든 준비가 완료되었습니다!

3️⃣ **서버 실행 (자동 모델 다운로드)**

```bash
python3 main_server.py
```

디버깅용:

```bash
# 프로젝트 내 모든 __pycache__ 디렉토리 및 .pyc 파일 삭제
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

PYTHONUNBUFFERED=1 python3 -B main_server.py
```

- 메인 스크립트 `main_server.py`를 실행하면 필요한 대용량 파일이 자동으로 Hugging Face에서 다운로드됩니다.
- 이미 존재하는 파일은 다시 다운로드하지 않습니다.
- 다운로드는 백그라운드에서 진행되므로 서버 시작에 지연은 없지만, 모든 다운로드가 완료된 후 각 기능을 사용하세요.

### 수동 다운로드

수동으로 파일을 다운로드하려면 다음 명령어를 실행하세요:

```bash
# 프로젝트 루트 디렉토리로 이동
cd storagy-repo-5

# 필요한 패키지 설치
pip install huggingface_hub

# Python 스크립트로 다운로드
python3 -c "
from huggingface_hub import hf_hub_download
import os

# 디렉토리 생성
os.makedirs('ai/checkpoint_a2b_inorm', exist_ok=True)
os.makedirs('ai/log_a2b_inorm/train', exist_ok=True)
os.makedirs('ai/log_a2b_inorm/val', exist_ok=True)
os.makedirs('SAM2_streaming/configs/sam2', exist_ok=True)

# 파일 다운로드
hf_hub_download(repo_id='YOUR_USERNAME/storagy-repo-5-models', filename='model_epoch400.pth', local_dir='ai/checkpoint_a2b_inorm')
hf_hub_download(repo_id='YOUR_USERNAME/storagy-repo-5-models', filename='events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.0', local_dir='ai/log_a2b_inorm/train')
hf_hub_download(repo_id='YOUR_USERNAME/storagy-repo-5-models', filename='events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.1', local_dir='ai/log_a2b_inorm/val')
hf_hub_download(repo_id='YOUR_USERNAME/storagy-repo-5-models', filename='sam2_hiera_tiny.pt', local_dir='SAM2_streaming/checkpoints/sam2')
"
```