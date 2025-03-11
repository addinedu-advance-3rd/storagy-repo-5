import os
import re
from flask import Blueprint, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import cv2, pytesseract
import numpy as np
from PIL import Image
import uuid
import time
import random
import yaml
import io
import traceback


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

# 메모리 내 파일 객체를 저장할 딕셔너리와 만료 시간 추적
MEMORY_FILES = {}
FILE_EXPIRY = {}  # 파일 만료 시간 추적
EXPIRY_TIME = 3600  # 1시간 후 만료

# 만료된 파일 정리 함수
def cleanup_expired_files():
    current_time = time.time()
    expired_keys = [k for k, v in FILE_EXPIRY.items() if current_time > v]
    for key in expired_keys:
        if key in MEMORY_FILES:
            del MEMORY_FILES[key]
        del FILE_EXPIRY[key]


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
            # processed_filename = f"{filename}.txt"
            processed_filename = "map.txt"
            processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
            with open(processed_filepath, 'w') as f:
                f.write(ocr_data)

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
        image_resized = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        gray = cv2.cvtColor(np.array(image_resized), cv2.COLOR_BGR2GRAY)
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


@guideline_bp.route('/generate', methods=['POST'])
def generate_files():
    """YAML 파일 및 PGM 파일 생성"""
    # YAML 파일 생성 위한 FlowList 클래스 정의 (YAML 리스트 포맷팅용)
    class FlowList(list):
        pass

    def flow_style_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    # 플로우 스타일 리스트 타입 등록
    yaml.add_representer(FlowList, flow_style_representer)

    # 응답 형식 지정
    generate_response = {'YAML': True, 'PGM': True, 'error': None}

    try:
        # 클라이언트에서 전송된 데이터 받기
        data = request.json
        
        # 사용자 키 가져오기 또는 생성
        user_key = request.cookies.get('user_key')
        if not user_key:
            user_key = str(uuid.uuid4())

        # 플로우 스타일 리스트로 변환
        origin_list = FlowList([
            float(data['origin']['x']), 
            float(data['origin']['y']), 
            float(data['origin']['theta'])
        ])
        
        # YAML 구조 설정
        yaml_data = {
            "image": "map.pgm",
            "mode": "trinary",  # (옵션) 3가지 색 구분 방식: 장애물, 이동 가능, 불확실
            "resolution": float(data['scale_factor']),  # 1 픽셀이 실제 세계에서 몇 m인지 설정 (m/pixel)
            "origin": origin_list,  # 맵의 원점: [X, Y, Theta]
            "occupied_thresh": 0.65,  # 점유(장애물) 임계값
            "free_thresh": 0.25,  # 자유 공간(이동 가능) 임계값
            "negate": 0  # 색상 반전 여부 (0이면 흰색=이동 가능, 1이면 흑색 반전)
        }
        # YAML 파일 생성 (메모리에만 저장)
        yaml_content = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
        yaml_file = io.BytesIO(yaml_content.encode('utf-8'))
        yaml_file.seek(0)

    except Exception as e:
        generate_response['YAML'] = False
        generate_response['error'] = "YAML 파일 생성 시 발생한 오류: \n" + str(e) + "\n\n"

    try:
        # PGM 파일 생성 (메모리에만 저장)
        # 실제 이미지 데이터가 있다면 이를 처리하여 PGM으로 변환
        # 여기서는 예시로 간단한 PGM 생성
        if 'image_data' in data and data['image_data']:
            # 이미지 데이터를 처리하는 코드 (실제 구현에 맞게 조정 필요)
            # 예: base64 디코딩된 이미지 데이터를 NumPy 배열로 변환 후 처리
            # image_data = process_image_data(data['image_data'])
            
            # 임시로 간단한 더미 PGM 만들기
            width, height = 100, 100  # 예시 크기
            pgm_header = f"P2\n{width} {height}\n255\n"
            pgm_data = "\n".join(" ".join(str(random.randint(0, 255)) for _ in range(width)) for _ in range(height))
            pgm_content = pgm_header + pgm_data
        else:
            # 간단한 PGM 파일 내용 생성 (데모용)
            pgm_content = "P2\n10 10\n255\n"
            for _ in range(10):
                row = " ".join(str(random.randint(0, 255)) for _ in range(10))
                pgm_content += row + "\n"
        
        pgm_file = io.BytesIO(pgm_content.encode('ascii'))
        pgm_file.seek(0)

        """
        # map.pgm 파일 생성 로직 (자리만 확보)
        # TODO: 실제 PGM 파일 생성 로직 구현
        # 원본 이미지를 처리하여 PGM 형식으로 변환 (여기서는 자리만 확보)
        pgm_path = os.path.join(PROCESSED_FOLDER, 'map.pgm')
        
        # 임시로 빈 PGM 파일 생성 (실제 구현 시 대체 필요)
        # 나중에 실제 pgm 생성 로직으로 대체할 것
        with open(pgm_path, 'w') as f:
            f.write('P2\n')  # PGM 헤더
            f.write('1 1\n')  # 너비 높이
            f.write('255\n')  # 최대 그레이스케일 값
            f.write('0\n')    # 빈 픽셀 값
        """

    except Exception as e:
        generate_response['PGM'] = False
        generate_response['error'] += "PGM 파일 생성 시 발생한 오류: \n" + str(e)

    # 메모리에 파일 객체 저장
    if user_key not in MEMORY_FILES:
        MEMORY_FILES[user_key] = {}
    if generate_response['YAML']:
        MEMORY_FILES[user_key]['yaml'] = yaml_file
    if generate_response['PGM']:
        MEMORY_FILES[user_key]['pgm'] = pgm_file
    # 만료 시간 설정 (현재 시간 + 1시간)
    FILE_EXPIRY[user_key] = time.time() + EXPIRY_TIME

    # 응답에 사용자 키 설정
    response = jsonify(generate_response)
    response.set_cookie('user_key', user_key, max_age=EXPIRY_TIME)
    return response


@guideline_bp.route('/download/yaml')
def download_yaml():
    try:
        user_key = request.cookies.get('user_key')
        if not user_key or user_key not in MEMORY_FILES or 'yaml' not in MEMORY_FILES[user_key]:
            return jsonify({"error": "YAML 파일이 생성되지 않았거나 만료되었습니다. 도면을 재업로드하세요."}), 404
        
        # 파일 사용 시 만료 시간 연장
        FILE_EXPIRY[user_key] = time.time() + EXPIRY_TIME
        
        yaml_file = MEMORY_FILES[user_key]['yaml']
        yaml_file.seek(0)
        
        # 로컬 리포지토리 내에 미리 저장
        yaml_path = os.path.join(PROCESSED_FOLDER, 'map.yaml')
        with open(yaml_path, 'wb') as f:  # 바이너리 모드로 열기
            f.write(yaml_file.getvalue())  # BytesIO의 내용을 가져와 쓰기

        # 파일 포인터를 다시 처음으로 이동
        yaml_file.seek(0)

        return send_file(
            yaml_file,
            as_attachment=True,
            download_name='map.yaml',
            mimetype='application/x-yaml'
        )
    except Exception as e:
        print("YAML 내보내기 오류:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@guideline_bp.route('/download/pgm')
def download_pgm():
    try:
        user_key = request.cookies.get('user_key')
        if not user_key or user_key not in MEMORY_FILES or 'pgm' not in MEMORY_FILES[user_key]:
            return jsonify({"error": "PGM 파일이 생성되지 않았거나 만료되었습니다. 도면을 재업로드하세요."}), 404
        
        # 파일 사용 시 만료 시간 연장
        FILE_EXPIRY[user_key] = time.time() + EXPIRY_TIME
        
        pgm_file = MEMORY_FILES[user_key]['pgm']
        pgm_file.seek(0)

        # 로컬 리포지토리 내에 미리 저장
        pgm_path = os.path.join(PROCESSED_FOLDER, 'map.pgm')
        with open(pgm_path, 'wb') as f:
            f.write(pgm_file.getvalue())

        # 파일 포인터를 다시 처음으로 이동
        pgm_file.seek(0)

        return send_file(
            pgm_file,
            as_attachment=True,
            download_name='map.pgm',
            mimetype='image/x-portable-graymap'  # 또는 'image/x-pgm'
        )
    except Exception as e:
        print("PGM 내보내기 오류:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500