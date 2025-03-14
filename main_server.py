from flask import Flask, render_template
from pages import tracking
from pages.guideline import guideline_bp
from pages.nav import nav_bp, init_nav
from pages.remap import remap_bp, init_remap
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import logging
import os, sys
from utils.download_large_files import download_models

# 프로젝트 루트(storagy-repo-5/)를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# pages.*.py의 Blueprint를 url_prefix 경로로 등록
app.register_blueprint(nav_bp, url_prefix='/nav')
app.register_blueprint(guideline_bp, url_prefix='/guideline')
app.register_blueprint(remap_bp, url_prefix='/remap')
app.register_blueprint(tracking.tracking_bp, url_prefix='/tracking')

@app.route('/')
def index():
    return render_template('main_server.html')

def check_required_files():
    """필요한 파일들이 존재하는지 확인하고 없으면 다운로드 시작"""
    files_to_check = [
        'SAM2_streaming/configs/sam2/sam2_hiera_tiny.pt',
        'ai/checkpoint_a2b_inorm/model_epoch400.pth',
        'ai/log_a2b_inorm/train/events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.0',
        'ai/log_a2b_inorm/val/events.out.tfevents.1741239033.addinedu-Bravo-17-D7VF.15745.1'
    ]
    
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    
    if missing_files:
        print(f"로컬 리포지토리에 없는 대용량 파일들을 다운로드합니다: {missing_files}")
        logger.info(f"로컬 리포지토리에 없는 대용량 파일들을 다운로드합니다: {missing_files}")
        # HuggingFace 공개 저장소에서 직접 다운로드
        download_models()
    else:
        print("필요한 모든 대용량 파일이 이미 있습니다. 다음 단계로 넘어갑니다.")
        logger.info("필요한 모든 대용량 파일이 이미 있습니다. 다음 단계로 넘어갑니다.")

if __name__ == '__main__':
    # 파일 확인 및 필요시 다운로드 시작
    check_required_files()

    # nav 관련 초기화 (맵 로드 및 ROS 스레드 시작)
    rclpy.init()

    # nav와 remap 노드 생성 (spin 호출 없음)
    nav_node = init_nav()      # Nav2Client 노드 생성 및 구독자 등록
    remap_node = init_remap()  # Nav2TrajectorySender 노드 생성

    # MultiThreadedExecutor 생성 후 두 노드 추가
    executor = MultiThreadedExecutor()
    executor.add_node(nav_node)
    executor.add_node(remap_node)

    # ROS executor를 별도 스레드에서 실행 (blocking하지 않도록)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    tracking.start_tracking()

    # Flask 웹 서버 실행 (메인 스레드)
    app.run(host="0.0.0.0", port=8000)

    # Flask 종료 후 ROS 종료
    executor.shutdown()
    nav_node.destroy_node()
    remap_node.destroy_node()
    rclpy.shutdown()