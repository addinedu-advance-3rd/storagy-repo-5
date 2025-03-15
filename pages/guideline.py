import os
import re
from flask import Blueprint, request, jsonify, render_template, url_for, send_file
from werkzeug.utils import secure_filename
import cv2, pytesseract
import numpy as np
from PIL import Image
import uuid
import time
import shutil
import yaml
import io
import traceback

from utils.config import UPLOADS_DIR, MAP_DIR, MAP_TEMP_DIR, ALLOWED_IMAGE_EXTENSIONS, AI_MODEL_DIR

from ai.main import get_parser
from ai.train import test

guideline_bp = Blueprint('guideline', __name__, template_folder='templates')

# 파일 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# 메모리 내 파일 객체를 저장할 딕셔너리와 만료 시간 추적
MEMORY_FILES = {}
FILE_EXPIRY = {}  # 파일 만료 시간 추적
EXPIRY_TIME = 3600  # 1시간

# 메모리 내 만료된 파일 정리 함수
def cleanup_expired_files():
    current_time = time.time()
    expired_keys = [k for k, v in FILE_EXPIRY.items() if current_time > v]
    for key in expired_keys:
        if key in MEMORY_FILES:
            del MEMORY_FILES[key]
        del FILE_EXPIRY[key]

# 이미지 OCR 처리 함수
def process_image(image):
    try:
        # OCR이 작은 텍스트나 기호도 추출할 수 있도록 
        # 이미지를 고해상도(300 DPI 이상)로 리샘플링하고 노이즈를 제거한다.
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
        print(f"❌ OCR 처리 중 오류 발생: {e}")
        return f"OCR 처리 중 오류 발생: {e}"

# OCR 결과 숫자 추출 함수
def extract_numbers(ocr_text):
    numbers = re.findall(r'\d+\.?\d*', ocr_text)  # 정규표현식으로 정수와 소수를 포함한 숫자 추출
    return [float(num) if '.' in num else int(num) for num in numbers]

# 도면 전처리 모듈 (ai/train.py) 실행을 위한 인자 생성 함수
def create_args(data_dir, result_dir, model_dir):
    """
    test 함수 실행을 위한 인자 객체 생성
    
    Args:
        input_file_path: 입력 이미지 경로
        output_dir: 출력 디렉토리
    
    Returns:
        argparse.Namespace: test 함수에 전달할 인자 객체
    """
    parser = get_parser()
    args = parser.parse_args()  # 빈 인자 목록 전달

    # mode를 test로 설정
    args.mode = "test"
    
    # 중요: data_dir를 직접 설정하여 /test 서브디렉토리를 사용하지 않도록 함
    args.data_dir = data_dir
    args.result_dir = result_dir
    args.ckpt_dir = model_dir

    return args

# (1) 도면 업로드 및 OCR 및 거리 측정 처리
@guideline_bp.route('/', methods=['GET', 'POST'])
def guideline():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "POST request에 파일 부분이 없습니다."}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "선택된 파일이 없습니다."}), 400

        if file and allowed_file(file.filename):
            # 파일 저장
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOADS_DIR, filename)
            file.save(filepath)

            # 원본 이미지 디렉토리 정리: 기존 파일 삭제 (도면 처리 AI는 input 디렉토리에 여러 파일이 있으면 에러 발생)
            if os.path.exists(MAP_TEMP_DIR):
                for f in os.listdir(MAP_TEMP_DIR):
                    p = os.path.join(MAP_TEMP_DIR, f)
                    if os.path.isfile(p):
                        os.remove(p)
            else:
                os.makedirs(MAP_TEMP_DIR)

            filepath_2 = os.path.join(MAP_TEMP_DIR, "map_original." + filename.rsplit('.', 1)[1].lower())
            shutil.copy2(filepath, filepath_2)

            # OCR 처리
            image = Image.open(filepath)
            ocr_data = process_image(image)

            # 처리된 데이터 저장
            processed_filename = "map.txt"
            processed_filepath = os.path.join(MAP_DIR, processed_filename)
            with open(processed_filepath, 'w') as f:
                f.write(ocr_data)

            # OCR 데이터에서 숫자 추출
            extracted_numbers = extract_numbers(ocr_data)

            # 정적 URL 경로로 변환 (/static/uploads/filename)
            relative_filepath = os.path.join('uploads', filename)

            # 업로드된 이미지와 OCR 데이터를 렌더링에 전달
            return render_template(
                'guideline.html',
                uploaded_image_url=url_for('static', filename=relative_filepath),
                ocr_data=ocr_data,
                extracted_numbers=extracted_numbers
            )

        return jsonify({"error": "잘못된 파일 형식입니다."}), 400

    # GET 요청 시 기본 페이지 렌더링
    return render_template(
        'guideline.html',
        uploaded_image_url=None,
        ocr_data=None,
        extracted_numbers=None
    )

# (2) YAML 파일 및 PGM 파일 생성
@guideline_bp.route('/generate', methods=['POST'])
def generate_files():
    # YAML 파일 생성 위한 FlowList 클래스 정의 (YAML 리스트 포맷팅용)
    class FlowList(list):
        pass

    def flow_style_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    # 플로우 스타일 리스트 타입 등록
    yaml.add_representer(FlowList, flow_style_representer)

    # 응답 형식 지정
    generate_response = {'YAML': True, 'PGM': True, 'error': ""}

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

        # YAML 파일 작성
        yaml_content = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
        yaml_memory = io.BytesIO(yaml_content.encode('utf-8'))
        yaml_memory.seek(0)

    except Exception as e:
        generate_response['YAML'] = False
        generate_response['error'] = "YAML 파일 생성 시 발생한 오류: \n" + str(e) + "\n\n"

    try:
        # 원본 이미지 파일 확인
        orig_files = [f for f in os.listdir(MAP_TEMP_DIR) if f.startswith('map_original')]

        if not orig_files:
            generate_response['PGM'] = False
            generate_response['error'] += "PGM 파일 생성 시 발생한 오류: \n" + str(e)
        else:
            args = create_args(MAP_TEMP_DIR, MAP_DIR, AI_MODEL_DIR)
            test(args)

            if "map.png" in [f for f in os.listdir(MAP_DIR)]:
                # 처리된 첫 번째 이미지 사용
                processed_filepath = os.path.join(MAP_DIR, "map.png")
                
                # 이미지를 메모리에 로드
                image = cv2.imread(processed_filepath)
                
                # 그레이스케일로 변환
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # 이진화
                # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                
                # PGM 데이터를 메모리에 생성
                height, width = binary.shape
                pgm_memory = io.BytesIO()
                
                # PGM 헤더 작성
                pgm_memory.write(b'P5\n')
                pgm_memory.write(f'{width} {height}\n'.encode())
                pgm_memory.write(b'255\n')
                
                # 이미지 데이터 작성
                pgm_memory.write(binary.tobytes())
                pgm_memory.seek(0)

    except Exception as e:
        generate_response['PGM'] = False
        generate_response['error'] += "PGM 파일 생성 시 발생한 오류: \n" + str(e)

    # 메모리에 YAML 파일 및 PGM 파일 객체 저장
    if user_key not in MEMORY_FILES:
        MEMORY_FILES[user_key] = {}
    if generate_response['YAML']:
        MEMORY_FILES[user_key]['yaml'] = yaml_memory
    if generate_response['PGM']:
        MEMORY_FILES[user_key]['pgm'] = pgm_memory
    # 만료 시간 설정 (현재 시간 + 1시간)
    FILE_EXPIRY[user_key] = time.time() + EXPIRY_TIME

    # 응답에 사용자 키 설정
    response = jsonify(generate_response)
    response.set_cookie('user_key', user_key, max_age=EXPIRY_TIME)
    return response

# (3-1) YAML 파일 다운로드
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
        
        # 로컬 리포지토리 내에 먼저 저장
        yaml_path = os.path.join(MAP_DIR, 'map.yaml')
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
        print("❌ YAML 내보내기 오류:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# (3-2) PGM 파일 다운로드
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

        # 로컬 리포지토리 내에 먼저 저장
        pgm_path = os.path.join(MAP_DIR, 'map.pgm')
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
        print("❌ PGM 내보내기 오류:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500