import torch
import numpy as np
import cv2
import socket
from sam2.build_sam import build_sam2_camera_predictor

# -------------------------------
# 1) CUDA & SAM2 설정
# -------------------------------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model_version = 'sam2'
sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_hiera_tiny.pt"
model_cfg = f"{model_version}/{model_version}_hiera_t.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# -------------------------------
# 2) UDP 설정 (이미지 수신 + 중심 좌표 송신)
# -------------------------------
UDP_IP = "0.0.0.0"
UDP_PORT_RGB = 5005            # (로봇 → 서버) 이미지를 받을 포트
ROBOT_IP = "192.168.1.12"       # 로봇 IP
UDP_PORT_CENTER = 5008         # (서버 → 로봇) 중심 좌표를 보낼 포트

# 소켓 생성 & 바인딩 (이미지 수신)
sock_rgb = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_rgb.bind((UDP_IP, UDP_PORT_RGB))

# 논블로킹 설정 → 프레임 스킵
sock_rgb.setblocking(False)

# 중심 좌표 송신 소켓
sock_center = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------------
# 전역 상태
# -------------------------------
point = None
point_selected = False
if_init = False
random_color = True
last_center_x = None
# -------------------------------
# 마우스 콜백
# -------------------------------
def on_mouse_click(event, x, y, flags, param):
    global point, point_selected
    if not point_selected and event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", on_mouse_click)

# -------------------------------
# 메인 루프 (단일 스레드)
# -------------------------------
while True:
    # (A) 네트워크에서 최신 프레임만 읽어오기 (프레임 스킵)
    last_frame_data = None
    while True:
        try:
            data, addr = sock_rgb.recvfrom(65536)
            last_frame_data = data  # 계속 갱신 → 최종적으로 가장 마지막 데이터만 남음
        except BlockingIOError:
            # 더 이상 수신할 데이터가 없음
            break
        except socket.timeout:
            # 타임아웃 발생
            break
        except Exception as e:
            print(f"❌ Error receiving RGB: {e}")
            break

    if last_frame_data is None:
        # 이번 루프에서 새 프레임 없음 → 이벤트 처리만
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # (B) 마지막 프레임 디코딩
    frame = cv2.imdecode(np.frombuffer(last_frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)  # 좌우 반전 (보기 편의상)
    
    # (C) SAM2로 객체 추적
    if not point_selected:
        # 객체 선택 안내
        cv2.putText(frame, "Select an object by clicking a point", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # 세그멘테이션
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

        # (D) 마스크 시각화 & 중심 좌표 계산 + 송신
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

            # 객체 중심
            if np.any(out_mask):
                M = cv2.moments(out_mask[:, :, 0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # ▶▶ 로봇으로 중심 좌표 송신
                    sock_center.sendto(f"{cx},{cy}".encode(), (ROBOT_IP, UDP_PORT_CENTER))
                    print(f"📤 Sent Center XY: {cx}, {cy}")
                    last_center_x = cx

                    # 시각화
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Center({cx},{cy})", (cx+10, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else: # 객체 검출 x 
                k = frame.shape[0]/3
                if last_center_x < k: # 마지막 검출이 왼쪽
                    cx,cy = 0,999
                elif k<=last_center_x<=2*k:
                    cx,cy = 999,0
                else:
                    cx,cy = 0,-999
                # ▶▶ 로봇으로 중심 좌표 송신
                sock_center.sendto(f"{cx},{cy}".encode(), (ROBOT_IP, UDP_PORT_CENTER))
                print(f"📤 Sent Center XY: {cx}, {cy}")

            # 마스크 오버레이
            frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

    # (E) 디스플레이 및 이벤트 처리
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cv2.destroyAllWindows()
sock_rgb.close()
sock_center.close()
