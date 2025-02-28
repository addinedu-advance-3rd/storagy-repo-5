'''통신으로 이미지, 포인트 받아서 queue(rgb,points_queue)에 저장


##스레드 1
rgb_queue에서 차례로 꺼내서 input으로
YOLO Detection
deepsort tracking
중심값 계산 -> 픽셀추출
==> 중심값이 중심에서 벗어난 만큼 cmd_vel 비율 조절


##스레드 2
points_queue에서 차례로 꺼내서 
중심 픽셀의 뎁스 받아오기 
==> 중심 픽셀의 뎁스에 따라 cmd_vel 세기 조절 
'''

from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#===============================DETECTION====================================

detector = Detector(classes = [0]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./weights/yolov7x.pt',) # pass the path to the trained weight file


# Pass in any image path or Numpy Image using 'BGR' format
result = detector.detect('./IO_data/input/images/leg1.png', plot_bb = False) # plot_bb = False output the predictions as [x,y,w,h, confidence, class]
if len(result.shape) == 3:# If it is image, convert it to proper image. detector will give "BGR" image
    result = Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)) 
image = cv2.imread('./IO_data/input/images/leg1.png')

for det in result:
    print('len(result) : ' ,len(result))
    print('det : ' , det)
    x, y, w, h, conf, cls = det
    # 좌표를 정수형으로 변환
    x, y, w, h = int(x), int(y), int(w), int(h)
    # 바운딩 박스 그리기
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 텍스트 출력: 클래스와 신뢰도
    text = f"{cls}: {conf:.2f}"
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # 결과 창에 시각화
print('image.shape : ', image.shape)
print('Detection is done')
# cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
# cv2.imshow("Detections", image)
# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
# cv2.destroyAllWindows()

#====================================ReID======================================================

# Initialise  class that binds detector and tracker in one class
print('start DeepSORT')
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)
frame = cv2.imread('./IO_data/input/images/leg1.png')
# tracker.update_frame(frame)
print("DeepSORT is done")

# 창 설정: Matplotlib를 interactive 모드로 설정
plt.ion()
fig, ax = plt.subplots()

while True:
    output_frame = tracker.update_frame(frame)
    if output_frame is None:
        print("이미지를 로드하지 못했습니다. 경로를 확인하세요.")
        break

    # 이미지 색상 형식 변환 (BGR -> RGB) - 만약 output_frame이 BGR이면 변환 필요
    output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    ax.clear()
    ax.imshow(output_rgb)
    ax.set_title("Tracking")
    plt.pause(1)  # 1초 대기. 필요에 따라 시간을 조절하세요.

    # 'q' 키 입력 체크 (키 입력은 Matplotlib 창에서는 직접 처리하기 어려우므로, 다른 방법을 사용해야 함)
    # 예를 들어, loop count로 임의 종료하거나, 별도의 이벤트 핸들러를 사용합니다.
    # 여기서는 간단하게 1초 대기 후 다음 프레임으로 넘어갑니다.

