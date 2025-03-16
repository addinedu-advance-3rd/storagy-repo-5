import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import LaserScan  # LaserScan 메시지 임포트
from geometry_msgs.msg import Twist
import socket
import numpy as np
import threading
import cv2


class CameraUDPServer(Node):
    def __init__(self):
        super().__init__('camera_udp_server')

        self.prev_linear = 0.0
        self.prev_angular = 0.0

        # (1) RGB 압축 이미지 구독 -> 리사이즈 -> UDP 전송
        self.rgb_subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.rgb_image_callback,
            10
        )

        # (2) CompressedDepth 구독
        self.compressed_depth_sub = self.create_subscription(
            CompressedImage,
            '/camera/depth/image_raw/compressedDepth',  # 실제 토픽명 확인
            self.compressed_depth_callback,
            10
        )

        # (2) LiDAR 데이터 구독
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',  
            self.lidar_callback,
            10
        )
        
        # (3) cmd_vel 발행
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # (4) UDP 소켓 설정
        self.udp_ip = "192.168.1.12"  # 서버 IP
        self.udp_port_rgb = 5010      # RGB 전송 포트
        self.udp_port_center = 5018   # 객체 중심 좌표 수신 포트

        self.sock_rgb = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_center = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_center.bind(("0.0.0.0", self.udp_port_center))

        self.cmd_vel_msg = Twist()

        self.latest_depth_image = None  # 압축Depth 디코딩 결과(16비트)
        self.get_logger().info("✅ UDP Server Started (RGB -> Server, Center_XY <- Server)")

        # (5) 객체 중심 좌표 수신 스레드 실행
        self.center_thread = threading.Thread(target=self.receive_center, daemon=True)
        self.center_thread.start()
        self.resize_w = 320
        self.resize_h = 200

        self.obstacle_detected = False
        self.small_x = 0
        self.small_y = 0
        self.front_min = 10000
    # -------------------------------------------------------
    # A. RGB 콜백 -> 리사이즈 후 서버로 전송
    # -------------------------------------------------------
    def rgb_image_callback(self, msg: CompressedImage): 
        """받은 RGB 압축 이미지를 리사이즈 후 서버로 UDP 전송"""
        try:
            # (1) CompressedImage -> np array
            np_arr = np.frombuffer(msg.data, np.uint8)
            original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if original_image is None:
                self.get_logger().warn("❌ Failed to decode compressed RGB image")
                return

            # (2) 리사이즈
            resized_image = cv2.resize(original_image, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

            # (3) JPEG로 재압축
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            success, encoded_image = cv2.imencode('.jpg', resized_image, encode_param)
            if not success:
                self.get_logger().warn("❌ Failed to re-encode resized image")
                return

            compressed_data = encoded_image.tobytes()

            # (4) UDP 전송
            self.sock_rgb.sendto(compressed_data, (self.udp_ip, self.udp_port_rgb))
            # self.get_logger().info(f"🟢 RGB Image sent (Resized: {self.resize_w}x{self.resize_h}, Size: {len(compressed_data)} bytes)")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to send RGB image: {e}")

    # -------------------------------------------------------
    # B. CompressedDepth 콜백
    # -------------------------------------------------------
    def compressed_depth_callback(self, msg: CompressedImage):
        try:
            raw = msg.data  # bytes
            # (1) 최소 12바이트 헤더가 앞에 존재
            if len(raw) <= 12:
                self.get_logger().warn("CompressedDepth data too small!")
                return 

            # (2) ROS 공식 플러그인 소스상, [8:12] 바이트가 실제 PNG 시작 오프셋을 담기도 함
            #    하지만 대부분 고정 12바이트로 잡히므로 여기서는 12로 가정
            offset = 12

            # (3) PNG 데이터만 추출
            png_data = raw[offset:]

            # (4) np.uint8 배열로 변환 후 imdecode
            np_arr = np.frombuffer(png_data, dtype=np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                self.get_logger().warn("❌ Failed to decode compressedDepth (PNG) after offset")
                return

            # depth_img.shape=(height, width), dtype=uint16 (보통 mm 단위)
            self.latest_depth_image = depth_img
            # self.get_logger().info(f"Decoded compressedDepth: shape={depth_img.shape}")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to decode compressedDepth: {e}")

    # -------------------------------------------------------
    # C. (center_x, center_y) 수신 스레드
    # -------------------------------------------------------
    def receive_center(self):

        

        """서버에서 보낸 객체 중심 좌표(center_x, center_y, 320x240 기준) 수신"""
        while True:
            
            try:
                data, addr = self.sock_center.recvfrom(1024)
                self.small_x, self.small_y = map(int, data.decode().split(',')) ## center

                if self.small_y == 999 : #  객체가 오른쪽에서 사라짐
                    self.cmd_vel_msg.linear.x = 0.0
                    self.cmd_vel_msg.angular.z = -0.8
                    self.cmd_vel_publisher.publish(self.cmd_vel_msg)
                    continue

                elif self.small_y ==-999: # 객체가 왼쪽에서 사라짐
                    self.cmd_vel_msg.linear.x = 0.0
                    self.cmd_vel_msg.angular.z = 0.8
                    self.cmd_vel_publisher.publish(self.cmd_vel_msg)
                    continue

                elif self.small_x == 999: # 객체와 카메라 중간에 장애물 
                    self.cmd_vel_msg.linear.x = self.prev_linear
                    self.cmd_vel_msg.angular.z = self.prev_angular


                else: #객체 존재   
                    # (1) 리사이즈 보정 → 실제 depth 이미지(640x400) 좌표
                    real_x = int(self.small_x * (640.0 / self.resize_w))  
                    real_y = int(self.small_y * (400.0 / self.resize_h))  

                    # (2) 압축Depth에서 (real_x, real_y) 픽셀의 깊이값(m) 얻기
                    depth_value_m = self.get_depth_from_compressed_depth(real_x, real_y)
         
                    if depth_value_m is not None or not np.isnan(depth_value_m):

                        # (3) cmd_vel 계산
                        # self.cmd_vel_msg = Twist()
                        angular_scale = 3
                        linear_scale = 0.1
                        angular_z = float(((self.small_x - self.resize_w/2) / self.resize_w/2))
                        if depth_value_m <= 0:  # 🚨 Depth 값이 비정상적으로 작은 경우, 기본값 1.0m로 설정
                            depth_value_m = 1.0

                        if 0<depth_value_m < 1 and 0<abs(angular_z) < 0.04:
                            # self.cmd_vel_msg.linear.x = max(0.05, depth_value_m * linear_scale * 0.5)
                            self.cmd_vel_msg.linear.x = 0.0
                            self.cmd_vel_msg.angular.z = 0.0
                            self.cmd_vel_publisher.publish(self.cmd_vel_msg)
                            continue

                        elif 0<depth_value_m < 1 and abs(angular_z) > 0.04:
                            # self.cmd_vel_msg.linear.x = max(0.05, depth_value_m * linear_scale * 0.5)
                            self.cmd_vel_msg.linear.x = 0.0
                            self.cmd_vel_msg.angular.z = angular_z * angular_scale
                            
                        elif depth_value_m > 1 and 0< abs(angular_z) < 0.04:
                            self.cmd_vel_msg.linear.x = float(depth_value_m * linear_scale)
                            self.cmd_vel_msg.angular.z = 0.0

                        else:
                            self.cmd_vel_msg.linear.x = float(depth_value_m * linear_scale)
                            self.cmd_vel_msg.angular.z = angular_z * angular_scale


                        


                # 🚀 부드러운 속도 조절 (Low-pass filtering)
                alpha = 0.2  # 부드러운 속도 조절 비율 (0~1 사이 값, 높을수록 빠르게 반응)
                self.cmd_vel_msg.linear.x = alpha * self.cmd_vel_msg.linear.x + (1 - alpha) * self.prev_linear
                self.cmd_vel_msg.angular.z = alpha * self.cmd_vel_msg.angular.z + (1 - alpha) * self.prev_angular

                # ✅ LiDAR 장애물 감지를 반영하여 이동 명령 조정
                self.adjust_cmd_vel(self.cmd_vel_msg.linear.x, self.cmd_vel_msg.angular.z)
                # 🔹 이전 속도값 갱신
                self.prev_linear = self.cmd_vel_msg.linear.x
                self.prev_angular = self.cmd_vel_msg.angular.z

                # self.cmd_vel_publisher.publish(self.cmd_vel_msg)

                
                # self.get_logger().info(f"Center=({self.small_x},{self.small_y}) -> Real=({real_x},{real_y}), depth={depth_value_m:.3f} m")
                # self.get_logger().info(f"🚀 Sent cmd_vel: lin.x={self.cmd_vel_msg.linear.x:.3f}, ang.z={self.cmd_vel_msg.angular.z:.3f}")

            except Exception as e:
                self.get_logger().error(f"❌ Failed to receive Center_XY: {e}")
        # -------------------------------------------------------
    # D. 압축Depth에서 (x, y) 픽셀의 깊이(m) 계산
    # -------------------------------------------------------
    def get_depth_from_compressed_depth(self, x, y):
        """
        latest_depth_image[y, x] = 16비트 값(mm) -> meter 변환
        """
        depth_raw = self.latest_depth_image[y, x]  # uint16 (대개 mm 단위)
        depth_m = depth_raw * 0.001  # mm -> m
        return float(depth_m)

    # -------------------------------------------------------
    # E. LiDAR 데이터 처리 (장애물 감지 및 회피)
    # -------------------------------------------------------
    def lidar_callback(self, msg: LaserScan):
        """LiDAR 데이터를 활용하여 장애물 감지 및 회피 로직 적용"""
        ranges = np.array(msg.ranges)  # LiDAR 거리 데이터 (리스트 -> numpy 배열 변환)
        valid_ranges = ranges[np.isfinite(ranges)]  # 유효한 거리 값만 선택
        if len(valid_ranges) == 0:
            return  # 데이터가 없으면 리턴

        min_distance = min(valid_ranges)  # 가장 가까운 장애물 거리
        front_ranges = ranges[len(ranges)//2 - 90 : len(ranges)//2+90]  # 전방 (약 ±90도)
        valid_front_ranges = [r for r in front_ranges if r>0.0]
        self.front_min = min(valid_front_ranges) if valid_front_ranges else None

        left_ranges = ranges[:len(ranges)//9]  # 왼쪽 영역 (약 -45도 ~ 0도)
        valid_left_ranges = [r for r in left_ranges if r>0.0]
        left_min = min(valid_left_ranges) if valid_left_ranges else None

        right_ranges = ranges[-len(ranges)//9:]  # 오른쪽 영역 (약 0도 ~ +45도)
        valid_right_ranges = [r for r in right_ranges if r>0.0]
        right_min = min(valid_right_ranges) if valid_right_ranges else None

        # 장애물 감지 여부
        self.obstacle_detected = self.front_min is not None and 0.03<self.front_min < 0.3
        self.left_blocked = left_min is not None and 0.03<left_min < 0.3  
        self.right_blocked = right_min is not None and 0.03<right_min < 0.3

        # 🔹 0.00m이 아니라 "No Data"로 출력하도록 변경
        self.front_min_display = f"{self.front_min:.2f}m" if self.front_min is not None else "No Data"
        left_min_display = f"{left_min:.2f}m" if left_min is not None else "No Data"
        right_min_display = f"{right_min:.2f}m" if right_min is not None else "No Data"

        self.get_logger().info(f"🔍 LiDAR: Front={self.front_min_display} | Left={left_min_display} | Right={right_min_display}")

    # -------------------------------------------------------
    # F. 장애물 회피 기반 cmd_vel 조정
    # -------------------------------------------------------
    def adjust_cmd_vel(self, linear_x, angular_z):
        """LiDAR 데이터를 고려하여 이동 속도 및 방향을 조정"""
        # self.cmd_vel_msg = Twist()

        if self.obstacle_detected:
            self.get_logger().warn("⚠️ 장애물 감지! 속도 조정 중...")

            # (1) 정면 장애물 감지 시 정지 또는 회전
            if self.left_blocked and not self.right_blocked:
                angular_z = -1.2 # 오른쪽으로 회피
            elif self.right_blocked and not self.left_blocked:
                angular_z = 1.2 # 왼쪽으로 회피
            elif (self.right_blocked and self.left_blocked) or self.front_min < 0.2:
                linear_x = 0.0
                angular_z = 0.0
            else:
                if (self.small_x - self.resize_w/2) > 0:
                    angular_z = 1.2
                else:
                    angular_z = -1.2

            linear_x = self.prev_linear  
        
        elif self.left_blocked:
            linear_x = self.prev_linear
            angular_z = -0.5

        
        elif self.right_blocked:
            linear_x = self.prev_linear
            angular_z = 0.5

        self.cmd_vel_msg.linear.x = linear_x
        self.cmd_vel_msg.angular.z = angular_z
        self.cmd_vel_publisher.publish(self.cmd_vel_msg)
        # self.get_logger().info(f"🚀 조정된 cmd_vel: lin.x={self.cmd_vel_msg.linear.x:.3f}, ang.z={self.cmd_vel_msg.angular.z:.3f}")


def main(args=None):
    rclpy.init(args=args)
    server = CameraUDPServer()
    rclpy.spin(server)
    server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


