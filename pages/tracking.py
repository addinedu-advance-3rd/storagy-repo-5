# pages/tracking.py

import torch
import numpy as np
import cv2
import socket
import threading
import time
from flask import Blueprint, Response, render_template, request, jsonify
import sys
import os

# 현재 파일 (pages/tracking.py)의 디렉토리를 기준으로 프로젝트 루트 계산
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# SAM2_streaming 폴더 경로 (프로젝트 루트/SAM2_streaming)
SAM2_STREAMING_DIR = os.path.join(BASE_DIR, "SAM2_streaming")

# SAM2 스트리밍 관련 모듈 경로 추가
sys.path.insert(0, SAM2_STREAMING_DIR)
from sam2.build_sam import build_sam2_camera_predictor

# Hydra 설정 파일 경로를 상대 경로로 지정 (프로젝트 루트/SAM2_streaming/configs)
os.environ["HYDRA_CONFIG_PATH"] = os.path.join(SAM2_STREAMING_DIR, "configs")

tracking_bp = Blueprint('tracking', __name__)

# -------------------------------
# 1) CUDA & SAM2 설정
# -------------------------------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model_version = 'sam2'
sam2_checkpoint = os.path.join(SAM2_STREAMING_DIR, "checkpoints", "sam2", "sam2_hiera_tiny.pt")
model_cfg = "sam2/sam2_hiera_t"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
# -------------------------------
# 2) UDP 설정 (이미지 수신 + 중심 좌표 송신)
# -------------------------------
UDP_IP = "0.0.0.0"
UDP_PORT_RGB = 5010            # (로봇 → 서버) 이미지를 받을 포트
ROBOT_IP = "192.168.1.4"       # 로봇 IP
UDP_PORT_CENTER = 5018         # (서버 → 로봇) 중심 좌표를 보낼 포트

sock_rgb = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_rgb.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_rgb.bind((UDP_IP, UDP_PORT_RGB))
sock_rgb.setblocking(False)

sock_center = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------------
# 전역 상태 변수
# -------------------------------
point = None           # 웹 클릭 좌표 (튜플: (x, y))
point_selected = False # 웹 클릭 이벤트 발생 여부
if_init = False        # SAM2 초기화 여부
random_color = True    # 마스크 컬러 랜덤 여부
last_center_x = None

# 처리된 최종 프레임을 저장할 변수 및 동기화용 락
latest_frame = None
frame_lock = threading.Lock()

# -------------------------------
# 메인 처리 루프 (백그라운드 스레드)
# -------------------------------
def process_frames():
    global latest_frame, point, point_selected, if_init, last_center_x

    while True:
        # (A) UDP에서 최신 프레임 읽어오기 (프레임 스킵)
        last_frame_data = None
        while True:
            try:
                data, addr = sock_rgb.recvfrom(65536)
                last_frame_data = data  # 계속 갱신하여 최종 프레임 사용
            except BlockingIOError:
                break
            except socket.timeout:
                break
            except Exception as e:
                print(f"❌ Error receiving RGB: {e}")
                break

        if last_frame_data is None:
            time.sleep(0.01)
            continue

        # (B) 프레임 디코딩 및 좌우 반전
        frame = cv2.imdecode(np.frombuffer(last_frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)

        # (C) SAM2 객체 추적 처리
        if not point_selected:
            cv2.putText(frame, "Select an object by clicking on the web page", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if not if_init:
                    if_init = True
                    predictor.load_first_frame(frame)
                    ann_frame_idx = 0
                    ann_obj_id = (1,)
                    labels = np.array([1], dtype=np.int32)
                    points = np.array([point], dtype=np.float32)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
                    )
                else:
                    out_obj_ids, out_mask_logits = predictor.track(frame)

            # (D) 마스크 시각화 및 중심 좌표 계산 + 로봇 송신
            if out_mask_logits is not None and len(out_mask_logits) > 0:
                out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                if random_color:
                    color = tuple(np.random.randint(0, 256, size=3))
                    colored_mask = np.zeros_like(frame, dtype=np.uint8)
                    for c in range(3):
                        colored_mask[:, :, c] = out_mask[:, :, 0] * color[c]
                else:
                    out_mask = out_mask * 255
                    colored_mask = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2RGB)

                if np.any(out_mask):
                    M = cv2.moments(out_mask[:, :, 0])
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        sock_center.sendto(f"{cx},{cy}".encode(), (ROBOT_IP, UDP_PORT_CENTER))
                        print(f"📤 Sent Center XY: {cx}, {cy}")
                        last_center_x = cx
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"Center({cx},{cy})", (cx+10, cy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    k = frame.shape[0] / 3
                    if last_center_x is not None and last_center_x < k:
                        cx, cy = 0, 999
                    elif last_center_x is not None and k <= last_center_x <= 2 * k:
                        cx, cy = 999, 0
                    else:
                        cx, cy = 0, -999
                    sock_center.sendto(f"{cx},{cy}".encode(), (ROBOT_IP, UDP_PORT_CENTER))
                    print(f"📤 Sent Center XY: {cx}, {cy}")

                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

        # (E) 최신 처리 프레임 저장 (웹 스트리밍용)
        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.01)

def generate_video_stream():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@tracking_bp.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@tracking_bp.route('/click', methods=['POST'])
def handle_click():
    global point, point_selected
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    point = (x, y)
    point_selected = True
    print(f"Received click at: ({x}, {y})")
    return jsonify({"message": f"Click received at ({x}, {y})"})

@tracking_bp.route('/reset', methods=['POST'])
def reset_tracking():
    global if_init, point_selected, point
    if_init = False
    point_selected = False
    point = None
    print("SAM2 tracking has been reset.")
    return jsonify({"message": "SAM2 tracking has been reset."})

@tracking_bp.route('/')
def tracking_index():
    return render_template('tracking.html')

def start_tracking():
    t = threading.Thread(target=process_frames, daemon=True)
    t.start()
    print("Tracking thread started.")
