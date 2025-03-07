import os
import re
from flask import Blueprint, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import cv2, pytesseract
import numpy as np
from PIL import Image
import yaml
import io
import textwrap

# Blueprint 생성
guideline_bp = Blueprint('guideline', __name__, template_folder='templates')

# 업로드 및 처리 디렉토리 설정
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    """파일 확장자 확인 함수"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@guideline_bp.route('/', methods=['GET', 'POST'])
def guideline():
    """도면 업로드 및 OCR 및 거리 측정 처리"""
    if request.method == 'POST':
        # 업로드된 파일 처리
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # 파일 저장
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # OCR 처리
            image = Image.open(filepath)
            ocr_data = process_image(image)

            # 처리된 데이터 저장
            processed_filename = f"{filename}.txt"
            processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
            with open(processed_filepath, 'w') as f:
                f.write(ocr_data)

            # OCR 결과를 Flask 터미널 로그에 출력
            print("OCR 결과:", ocr_data)
            """
            터미널에 출력되지 않는 경우: 
            Python은 기본적으로 출력 버퍼링을 사용하기에, 이를 비활성화하면 로그가 바로 출력된다.
            Python 스크립트를 실행할 때 PYTHONUNBUFFERED 환경 변수를 설정하여 
            출력 버퍼링을 비활성화할 수 있다.
            PYTHONUNBUFFERED=1 python3 main_server.py
            """

            # OCR 데이터에서 숫자 추출
            extracted_numbers = extract_numbers(ocr_data)

            # 업로드된 이미지와 OCR 데이터를 렌더링에 전달
            return render_template(
                'guideline.html',
                uploaded_image_url=f'/static/uploads/{filename}',
                ocr_data=ocr_data,
                extracted_numbers=extracted_numbers
            )

        return jsonify({"error": "Invalid file type"}), 400

    # GET 요청 시 기본 페이지 렌더링
    return render_template(
        'guideline.html',
        uploaded_image_url=None,
        ocr_data=None,
        extracted_numbers=None
    )

def process_image(image):
    """이미지 OCR 처리 함수"""
    try:
        """
        OCR이 작은 텍스트나 기호도 추출할 수 있도록 
        이미지를 고해상도(300 DPI 이상)로 리샘플링하고 노이즈를 제거한다.
        """
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_image = cv2.medianBlur(binary, 3)
        
        # Tesseract로 OCR 수행
        # --oem: OCR 엔진 모드 (3: LSTM, 1: Legacy), --psm: 페이지 세그멘테이션 모드, c: 추가 설정
        # https://stackoverflow.com/questions/49587228/pytesseract-tessedit-char-whitelist-not-accepting-quote
        # config = """--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789.,`/\\"\\' """
        # config 옵션 없이 기본값 사용이 결과가 제일 잘 나왔다.
        ocr_result = pytesseract.image_to_string(processed_image, config=None, lang='eng')

        return ocr_result
    
    except Exception as e:

        return f"OCR 처리 중 오류 발생: {e}"

def extract_numbers(ocr_text):
    """OCR 결과에서 숫자 추출"""
    # 정규표현식을 사용하여 숫자 추출
    numbers = re.findall(r'\d+\.?\d*', ocr_text)  # 정수와 소수를 포함한 숫자 추출
    return [float(num) if '.' in num else int(num) for num in numbers]


@guideline_bp.route('/export_yaml', methods=['POST'])
def export_yaml():
    """사용자 데이터를 YAML 파일로 내보내기"""
    class FlowList(list):
        pass

    def flow_style_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    # 플로우 스타일 리스트 타입 등록
    yaml.add_representer(FlowList, flow_style_representer)

    try:
        # 클라이언트에서 전송된 데이터 받기
        data = request.json
        
        # 플로우 스타일 리스트로 변환
        origin_list = FlowList([
            float(data['origin']['x']), 
            float(data['origin']['y']), 
            float(data['origin']['theta'])
        ])
        
        # YAML 구조 설정
        yaml_data = {
            "image": "map.pgm",
            "mode": "trinary",  # (옵션) 3가지 색 구분 방식 (장애물, 이동 가능, 불확실)
            "resolution": float(data['scale_factor']),  # 1 픽셀이 실제 세계에서 몇 m인지 설정 (m/pixel)
            "origin": origin_list,  # 맵의 원점 (X, Y, Theta)
            "occupied_thresh": 0.65,  # 점유(장애물) 임계값
            "free_thresh": 0.25,  # 자유 공간(이동 가능) 임계값
            "negate": 0  # 색상 반전 여부 (0이면 흰색=이동 가능, 1이면 흑색 반전)
        }
        # YAML 파일 생성
        yaml_content = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
        
        # 메모리 내 파일로 변환
        yaml_file = io.BytesIO(yaml_content.encode('utf-8'))
        yaml_file.seek(0)
        
        # 파일 다운로드 제공
        return send_file(
            yaml_file,
            as_attachment=True,
            download_name='map.yaml',
            mimetype='application/x-yaml'
        )
    
    except Exception as e:
        import traceback
        print("YAML 내보내기 오류:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500