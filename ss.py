import cv2
import numpy as np
from detection_helpers import Detector
from tracking_helpers import *
from bridge_wrapper import *  # 이미 임포트되어 있다면 중복 선언 제거
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt

def main():
    # 1) 웹캠 설정
    cap = cv2.VideoCapture(0)  # 기본 웹캠. 장치 번호는 상황에 따라 조정 가능
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    # 2) YOLO & Deep SORT 초기화
    print("✅ Loading YOLO and DeepSORT...")
    detector = Detector(classes=[0])  # 예: 사람 클래스만 검출
    detector.load_model('./weights/yolov7.pt')
    tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb",
                              detector=detector)

    print("✅ YOLO + DeepSORT initialized.")

    # 3) 메인 루프: 카메라 프레임을 받아 YOLO + DeepSORT 추적
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        # 3-1) YOLO 검출 (bounding boxes 등)
        # plot_bb=False → (N, 6) 형태로 [x, y, w, h, conf, class] 반환
        detections = detector.detect(frame, plot_bb=False)
        if detections is None:
            # 검출된 객체가 없으면 그냥 웹캠만 표시
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 3-2) DeepSORT 추적 업데이트
        # tracker.update_frame() 내부적으로 YOLO 검출 결과 재가공 + Tracking 수행
        tracked_frame = tracker.update_frame(frame)

        # 3-3) 결과 표시
        cv2.imshow("Webcam", tracked_frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
