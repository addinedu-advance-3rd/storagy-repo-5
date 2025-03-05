from flask import Flask, render_template, render_template_string
import tracking
import guideline
from nav import nav_bp, init_nav

app = Flask(__name__)

# nav.py Blueprint를 '/nav' 경로로 등록
app.register_blueprint(nav_bp, url_prefix='/nav')

@app.route('/')
def index():
    return render_template('main_server.html')

@app.route('/tracking')
def tracking_route():
    tracking.start_tracking()
    return "Tracking function triggered."

@app.route('/guideline')
def guideline_route():
    guideline.start_guideline()
    return "Guideline function triggered."

if __name__ == '__main__':
    # nav 관련 초기화 (맵 로드 및 ROS 스레드 시작)
    init_nav()
    app.run(host="0.0.0.0", port=8000)
