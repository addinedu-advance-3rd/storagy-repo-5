import socket
import cv2
import numpy as np
import threading
import queue
from sensor_msgs.msg import PointCloud2
from rclpy.serialization import deserialize_message
from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'

# UDP 수신 설정
UDP_IP = "0.0.0.0"
UDP_PORT_RGB = 5005        # RGB 전송 포트
UDP_PORT_POINTS = 5006     # 포인트클라우드 전송 포트
BUFFER_SIZE = 65535

#Load YOLO,DeepSORT model
detector = Detector(classes = [0]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./weights/yolov7.pt',) # pass the path to the trained weight file
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# UDP 소켓 생성 및 바인딩
sock_rgb = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_rgb.bind((UDP_IP, UDP_PORT_RGB))

sock_points = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_points.bind((UDP_IP, UDP_PORT_POINTS))

print(f"✅ UDP Receiver Started, listening on RGB:{UDP_PORT_RGB} and PointCloud:{UDP_PORT_POINTS}")

# 데이터 저장 큐
rgb_queue = queue.Queue()
# 포인트클라우드 재조립을 위한 딕셔너리 (여기서는 한 메시지씩 처리한다고 가정)
pointcloud_packets = {}

def receive_rgb():
    """ RGB 이미지 수신 및 OpenCV 창에 표시 """
    while True:
        try:
            data, addr = sock_rgb.recvfrom(BUFFER_SIZE)
            print(f"RGB: Received {len(data)} bytes from {addr}")
            np_arr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is not None:
                rgb_queue.put(image)
            else:
                print("RGB: cv2.imdecode returned None")
        except Exception as e:
            print(f"❌ Error receiving RGB: {e}")

def receive_points():
    """ PointCloud 데이터 수신 및 재조립 """
    global pointcloud_packets
    while True:
        try:
            data, addr = sock_points.recvfrom(BUFFER_SIZE)
            # 패킷 형식: header|payload, header는 예: b"i/num"
            if b'|' not in data:
                print("PointCloud: Missing header separator")
                continue
            header, payload = data.split(b'|', 1)
            try:
                header_str = header.decode('utf-8')
                index_str, total_str = header_str.split('/')
                index = int(index_str)
                total = int(total_str)
            except Exception as e:
                print(f"PointCloud: Failed to parse header: {e}")
                continue

            # 재조립: 여기서는 한 번에 하나의 메시지만 처리한다고 가정
            # (실제 구현에서는 고유 메시 ID를 이용해야 함)
            pointcloud_packets[index] = payload
            # print(f"PointCloud: Received packet {index+1}/{total} from {addr}")
            if len(pointcloud_packets) == total:
                # 모든 패킷 수신 완료 → 재조립
                assembled = b''.join(pointcloud_packets[i] for i in range(total))
                try:
                    msg = deserialize_message(assembled, PointCloud2)
                    print(f"PointCloud Received: width={msg.width}, height={msg.height}, "
                          f"point_step={msg.point_step}, row_step={msg.row_step}")
                except Exception as e:
                    print(f"PointCloud: Failed to deserialize: {e}")
                # 초기화
                pointcloud_packets = {}
        except Exception as e:
            print(f"❌ Error receiving PointCloud: {e}")

# 스레드 시작
thread_rgb = threading.Thread(target=receive_rgb, daemon=True)
thread_points = threading.Thread(target=receive_points, daemon=True)
thread_rgb.start()
thread_points.start()

while True:
    if not rgb_queue.empty():
        image = rgb_queue.get()
        ## YOLO ##
        result = detector.detect(image, plot_bb = False) # plot_bb = False output the predictions as [x,y,w,h, confidence, class]
        
        if result is None:
            continue

        if len(result.shape) == 3:# If it is image, convert it to proper image. detector will give "BGR" image
            print('111111111111111')
            result = Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
        
        for det in result:
            print('len(result) : ' ,len(result))
            print('det : ' , det)

            # #시각화##
            # try:
            #     x, y, w, h, _, _ = det
            # except Exception as e:
            #     print("Error unpacking detection:", e)
            #     continue
            # # 좌표를 정수형으로 변환
            # x, y, w, h = int(x), int(y), int(w), int(h)
            # # 바운딩 박스 그리기
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        output_frame = tracker.update_frame(image)

        cv2.imshow("RGB Image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

sock_rgb.close()
sock_points.close()
cv2.destroyAllWindows()

# import socket
# import cv2
# import numpy as np
# import threading
# import queue
# from sensor_msgs.msg import PointCloud2
# from rclpy.serialization import deserialize_message

# # UDP 수신 설정
# UDP_IP = "0.0.0.0"
# UDP_PORT_RGB = 5005        # RGB 전송 포트
# UDP_PORT_POINTS = 5006     # 포인트클라우드 전송 포트
# BUFFER_SIZE = 65535

# # UDP 소켓 생성 및 바인딩
# sock_rgb = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock_rgb.bind((UDP_IP, UDP_PORT_RGB))

# sock_points = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock_points.bind((UDP_IP, UDP_PORT_POINTS))

# print(f"✅ UDP Receiver Started, listening on RGB:{UDP_PORT_RGB} and PointCloud:{UDP_PORT_POINTS}")

# # 데이터 저장 큐 (이미지 표시를 위한)
# rgb_queue = queue.Queue()

# # 분할 패킷 재조립을 위한 딕셔너리
# # 실제 구현 시, 프레임마다 고유 ID가 있어야 다중 프레임을 구분 가능하지만,
# # 여기서는 "한 번에 한 메시지"만 처리한다고 가정.
# rgb_packets = {}
# pointcloud_packets = {}

# def receive_rgb():
#     """ 분할 전송된 RGB 이미지 수신 및 재조립 """
#     global rgb_packets
#     while True:
#         try:
#             data, addr = sock_rgb.recvfrom(BUFFER_SIZE)
#             # 'i/num'|payload 형태
#             if b'|' not in data:
#                 print("RGB: Missing header separator")
#                 continue
#             header, payload = data.split(b'|', 1)
#             try:
#                 header_str = header.decode('utf-8')
#                 index_str, total_str = header_str.split('/')
#                 index = int(index_str)
#                 total = int(total_str)
#             except Exception as e:
#                 print(f"RGB: Failed to parse header: {e}")
#                 continue

#             rgb_packets[index] = payload
#             print(f"RGB: Received packet {index+1}/{total} from {addr}")
            
#             if len(rgb_packets) == total:
#                 # 모든 패킷 수신 완료 → 재조립
#                 assembled = b''.join(rgb_packets[i] for i in range(total))
#                 rgb_packets = {}  # 다음 프레임 대비 초기화

#                 # assembled가 최종 바이트 배열
#                 image = cv2.imdecode(np.frombuffer(assembled, np.uint8), cv2.IMREAD_COLOR)
#                 if image is not None:
#                     rgb_queue.put(image)
#                 else:
#                     print("RGB: cv2.imdecode returned None (reassembly failed)")
#         except Exception as e:
#             print(f"❌ Error receiving RGB: {e}")

# def receive_points():
#     """ PointCloud 데이터 수신 및 재조립 """
#     global pointcloud_packets
#     while True:
#         try:
#             data, addr = sock_points.recvfrom(BUFFER_SIZE)
#             # 패킷 형식: header|payload, header는 예: b"i/num"
#             if b'|' not in data:
#                 print("PointCloud: Missing header separator")
#                 continue
#             header, payload = data.split(b'|', 1)
#             try:
#                 header_str = header.decode('utf-8')
#                 index_str, total_str = header_str.split('/')
#                 index = int(index_str)
#                 total = int(total_str)
#             except Exception as e:
#                 print(f"PointCloud: Failed to parse header: {e}")
#                 continue

#             pointcloud_packets[index] = payload
#             print(f"PointCloud: Received packet {index+1}/{total} from {addr}")
            
#             if len(pointcloud_packets) == total:
#                 # 모든 패킷 수신 완료 → 재조립
#                 assembled = b''.join(pointcloud_packets[i] for i in range(total))
#                 pointcloud_packets = {}  # 초기화
#                 try:
#                     msg = deserialize_message(assembled, PointCloud2)
#                     print(f"PointCloud Received: width={msg.width}, height={msg.height}, "
#                           f"point_step={msg.point_step}, row_step={msg.row_step}")
#                 except Exception as e:
#                     print(f"PointCloud: Failed to deserialize: {e}")
#         except Exception as e:
#             print(f"❌ Error receiving PointCloud: {e}")

# # 스레드 시작
# thread_rgb = threading.Thread(target=receive_rgb, daemon=True)
# thread_points = threading.Thread(target=receive_points, daemon=True)
# thread_rgb.start()
# thread_points.start()

# # 메인 루프: RGB 이미지를 화면에 표시
# while True:
#     if not rgb_queue.empty():
#         image = rgb_queue.get()
#         cv2.imshow("RGB Image", image)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# sock_rgb.close()
# sock_points.close()
# cv2.destroyAllWindows()
