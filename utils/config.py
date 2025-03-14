import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, 'uploads')
MAP_DIR = os.path.join(STATIC_DIR, 'map')
MAP_TEMP_DIR = os.path.join(MAP_DIR, 'temp')
DEBUG_IMG_DIR = os.path.join(STATIC_DIR, "debug")

YAML_PATH = os.path.join(MAP_DIR, "map.yaml")
MAP_PNG_PATH = os.path.join(MAP_DIR, "map.png")
MAP_PGM_PATH = os.path.join(MAP_DIR, "map.pgm")
MAP_TXT_PATH = os.path.join(MAP_DIR, "map.txt")

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# 필요한 디렉토리 생성
def ensure_directories_exist():
    directories = [
        STATIC_DIR,
        UPLOADS_DIR,
        MAP_DIR,
        MAP_TEMP_DIR,
        DEBUG_IMG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"디렉토리 확인: {directory}")

# 파일이 임포트될 때 디렉토리 생성 실행
ensure_directories_exist()
