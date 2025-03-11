# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, 'uploads')
MAP_DIR = os.path.join(STATIC_DIR, 'map')
DEBUG_IMG_DIR = os.path.join(STATIC_DIR, "debug")

YAML_PATH = os.path.join(MAP_DIR, "map.yaml")
MAP_PNG_PATH = os.path.join(MAP_DIR, "map.png")
MAP_PGM_PATH = os.path.join(MAP_DIR, "map.pgm")
MAP_TXT_PATH = os.path.join(MAP_DIR, "map.txt")

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}