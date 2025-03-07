import os
import re
from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2, pytesseract
import numpy as np
from PIL import Image

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