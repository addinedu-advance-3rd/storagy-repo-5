from flask import Flask, render_template, render_template_string
from pages import tracking
from pages.guideline import guideline_bp
from pages.nav import nav_bp, init_nav
from pages.remap import remap_bp, init_remap
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading

app = Flask(__name__)

# pages.*.py의 Blueprint를 url_prefix 경로로 등록
app.register_blueprint(nav_bp, url_prefix='/nav')
app.register_blueprint(guideline_bp, url_prefix='/guideline')
app.register_blueprint(remap_bp, url_prefix='/remap')

@app.route('/')
def index():
    return render_template('main_server.html')

@app.route('/tracking')
def tracking_route():
    tracking.start_tracking()
    return "Tracking function triggered."

if __name__ == '__main__':

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

    # Flask 웹 서버 실행 (메인 스레드)
    app.run(host="0.0.0.0", port=8000)

    # Flask 종료 후 ROS 종료
    executor.shutdown()
    nav_node.destroy_node()
    remap_node.destroy_node()
    rclpy.shutdown()