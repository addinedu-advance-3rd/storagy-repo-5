# storagy-repo-5

### 자동 모델 다운로드 기능

이 프로젝트는 자동으로 필요한 대용량 모델 및 로그 파일을 다운로드합니다:

- `main_server.py`를 실행하면 필요한 파일이 자동으로 Hugging Face에서 다운로드됩니다.
- 이미 존재하는 파일은 다시 다운로드하지 않습니다.
- 다운로드는 백그라운드에서 진행되므로 서버 시작에 지연이 없습니다.

### 수동 다운로드

수동으로 파일을 다운로드하려면 다음 명령어를 실행하세요:

```bash
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
hf_hub_download(repo_id='YOUR_USERNAME/storagy-repo-5-models', filename='train_logs/events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.0', local_dir='ai/log_a2b_inorm/train')
hf_hub_download(repo_id='YOUR_USERNAME/storagy-repo-5-models', filename='val_logs/events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.1', local_dir='ai/log_a2b_inorm/val')
hf_hub_download(repo_id='YOUR_USERNAME/storagy-repo-5-models', filename='SAM2_streaming/configs/sam2/sam2_hiera_tiny.pt', local_dir='SAM2_streaming/configs/sam2')
"