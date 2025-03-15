import os
import logging
import subprocess
import threading
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 고정된 리포지토리 정보
HF_REPO_ID = "Supernova0417/storagy-repo-5-models"

class HFDownloader:
    """Hugging Face에서 대용량 파일을 다운로드하는 유틸리티 클래스"""
    
    def __init__(self):
        """초기화"""
        self.repo_id = HF_REPO_ID
        # 향후 새로운 파일이 추가되면 아래 리스트에 항목을 추가
        self.files_to_download = [
            {
                'remote_path': 'model_epoch400.pth',
                'local_path': 'ai/checkpoint_a2b_inorm/model_epoch400.pth'
            },
            {
                'remote_path': 'events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.0',
                'local_path': 'ai/log_a2b_inorm/train/events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.0'
            },
            {
                'remote_path': 'events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.1',
                'local_path': 'ai/log_a2b_inorm/val/events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.1'
            },
            {
                'remote_path': 'sam2_hiera_tiny.pt',
                'local_path': 'SAM2_streaming/configs/sam2/sam2_hiera_tiny.pt'
            }
        ]
    
    def ensure_hf_hub_installed(self):
        """huggingface_hub 패키지가 설치되어 있는지 확인"""
        try:
            import huggingface_hub
            return True
        except ImportError:
            logger.info("huggingface_hub 패키지 설치 중...")
            try:
                subprocess.check_call(["pip", "install", "-q", "huggingface_hub"])
                return True
            except Exception as e:
                logger.error(f"huggingface_hub 설치 실패: {str(e)}")
                return False
    
    def download_file(self, remote_path, local_path):
        """
        Hugging Face에서 파일 다운로드
        
        Args:
            remote_path: Hugging Face 리포지토리 내 파일 경로
            local_path: 로컬 저장 경로
        
        Returns:
            bool: 다운로드 성공 여부
        """
        if os.path.exists(local_path):
            logger.info(f"이미 파일이 존재합니다: {local_path}")
            return True
            
        try:
            from huggingface_hub import hf_hub_download
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 파일 다운로드
            logger.info(f"{self.repo_id}로부터 {remote_path} 다운로드 중...")
            download_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                local_dir=os.path.dirname(local_path),
                local_dir_use_symlinks=False
            )
            
            # 필요시 파일 이름 변경
            if os.path.basename(download_path) != os.path.basename(local_path):
                os.rename(download_path, local_path)
                
            logger.info(f"{local_path}에 성공적으로 다운로드되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"{remote_path} 다운로드 중 에러 발생: {str(e)}")
            return False
    
    def download_all_files(self):
        """모든 필요 파일 다운로드"""
        if not self.ensure_hf_hub_installed():
            logger.error("huggingface_hub 패키지 설치 확인에 실패했습니다.")
            return False
            
        success = True
        for file_info in self.files_to_download:
            if not self.download_file(file_info['remote_path'], file_info['local_path']):
                success = False
                
        return success
    
    def download_all_files_async(self, callback=None):
        """
        백그라운드 스레드에서 모든 파일 비동기 다운로드
        
        Args:
            callback: 다운로드 완료 후 호출할 콜백 함수 (선택 사항)
        """
        def download_thread():
            result = self.download_all_files()
            if callback:
                callback(result)
                
        thread = threading.Thread(target=download_thread)
        thread.daemon = True
        thread.start()
        return thread

def download_models():
    """
    메인 다운로드 함수 - 사용자명 파라미터 제거
    """
    downloader = HFDownloader()
    
    def on_download_complete(success):
        if success:
            print("성공적으로 모든 대용량 파일을 다운로드했습니다!")
            logger.info("성공적으로 모든 대용량 파일을 다운로드했습니다!")
        else:
            print("몇몇 대용량 파일을 다운로드 받는 데 실패했습니다. 로그를 확인하세요.")
            logger.warning("몇몇 대용량 파일을 다운로드 받는 데 실패했습니다. 로그를 확인하세요.")
    
    # 비동기 다운로드 시작
    downloader.download_all_files_async(callback=on_download_complete)
    print(f"{HF_REPO_ID}로부터 대용량 파일을 비동기 다운로드 중...")
    logger.info(f"Model downloads started in background from {HF_REPO_ID}...")

download_models()