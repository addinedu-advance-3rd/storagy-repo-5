from flask import Flask, render_template, render_template_string
import tracking
from guideline import guideline_bp
from nav import nav_bp, init_nav

app = Flask(__name__)

# nav.py Blueprint를 '/nav' 경로로 등록
app.register_blueprint(nav_bp, url_prefix='/nav')
# guideline.py Blueprint를 '/guideline' 경로로 등록
app.register_blueprint(guideline_bp, url_prefix='/guideline')

@app.route('/')
def index():
    return render_template('main_server.html')

@app.route('/tracking')
def tracking_route():
    tracking.start_tracking()
    return "Tracking function triggered."

if __name__ == '__main__':
    # nav 관련 초기화 (맵 로드 및 ROS 스레드 시작)
    init_nav()
    app.run(host="0.0.0.0", port=8000)
