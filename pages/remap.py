import os
import threading
import time
import math
import yaml
import cv2
import re
import heapq
import subprocess
from flask import Flask, jsonify, render_template_string, send_file, request, Blueprint
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from action_msgs.msg import GoalStatus

from robot.path_planning import generate_trajectory
from config import YAML_PATH, MAP_PNG_PATH

remap_bp = Blueprint('remap', __name__, template_folder='templates')

# 전역 변수 (ROS2와 Flask 간 데이터 공유)
current_pose_data = None
robot_path_data = []
total_waypoints = 0
remaining_waypoints = 0
trajectory_points = []  # trajectory.txt에 기록된 모든 waypoint 좌표

# -------------------------------------------
# Flask용: YAML과 PGM 파일을 읽어 PNG 맵 이미지 생성
# -------------------------------------------
def load_map_image():
    if not os.path.exists(YAML_PATH):
        print("❌ YAML 파일을 찾을 수 없습니다.!!")
        return None, None
    with open(YAML_PATH, 'r') as file:
        map_config = yaml.safe_load(file)
    pgm_path = os.path.join(os.path.dirname(YAML_PATH), map_config["image"])
    if not os.path.exists(pgm_path):
        print(f"❌ PGM 파일을 찾을 수 없습니다: {pgm_path}")
        return None, None
    pgm_image = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
    height, width = pgm_image.shape[:2]

    cv2.imwrite(MAP_PNG_PATH, pgm_image)
    print(f"remapping PNG 맵 생성 완료: {MAP_PNG_PATH}")
    
    return MAP_PNG_PATH, {"resolution": map_config["resolution"],
                      "origin": map_config["origin"],
                      "width": width,
                      "height":  height}

# -------------------------------------------
# ROS2 노드: Nav2TrajectorySender (경로 전송 및 /amcl_pose 구독)
# -------------------------------------------
class Nav2TrajectorySender(Node):
    def __init__(self):
        global total_waypoints, remaining_waypoints, trajectory_points

        super().__init__('nav2_trajectory_sender')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_data = self.load_map_yaml()
        self.trajectory_points = self.load_trajectory()
        self.current_index = 0
        self.revisited_paths = set()
        
        total_waypoints = len(self.trajectory_points)
        remaining_waypoints = total_waypoints - self.current_index
        trajectory_points = self.trajectory_points  # 전역 변수에 waypoint 좌표 저장
        
        # /amcl_pose 토픽을 구독하여 현재 위치 업데이트
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        
        if not self.trajectory_points:
            self.get_logger().info("❌ trajectory 좌표를 찾을 수 없습니다.")

        # 장애물 확인
        self.latest_costmap = None
        self.costmap_sub = self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, 10)

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def check_goal_in_obstacle(self, goal_x, goal_y):
        if self.latest_costmap is None:
            print("latest_costmap is none")
            return False

        resolution = self.latest_costmap.info.resolution
        width = self.latest_costmap.info.width
        height = self.latest_costmap.info.height
        origin_x = self.latest_costmap.info.origin.position.x
        origin_y = self.latest_costmap.info.origin.position.y

        goal_cell_x = int((goal_x - origin_x) / resolution)
        goal_cell_y = int((goal_y - origin_y) / resolution)

        if 0 <= goal_cell_x < width and 0 <= goal_cell_y < height:
            index = goal_cell_y * width + goal_cell_x
            print("해당 위치 장애물 0~100 사이의 값: ",self.latest_costmap.data[index])
            if self.latest_costmap.data[index] > 50:  # 50 이상이면 장애물
                return True
        return False

    def load_map_yaml(self):
        if not os.path.exists(YAML_PATH):
            self.get_logger().error("❌ YAML 파일을 찾을 수 없습니다.~~")
            return None
        with open(YAML_PATH, 'r') as file:
            map_config = yaml.safe_load(file)
        pgm_path = os.path.join(os.path.dirname(YAML_PATH), map_config["image"])
        if not os.path.exists(pgm_path):
            self.get_logger().error(f"❌ PGM 파일을 찾을 수 없습니다: {pgm_path}")
            return None
        pgm_image = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
        height, width = pgm_image.shape[:2]
        self.get_logger().info(f"📂 PGM 파일 로드 완료: {pgm_path}, 너비={width}, 높이={height}")
        return {"resolution": map_config["resolution"],
                "origin": map_config["origin"],
                "width": width,
                "height": height}

    def pixel_to_world(self, x_pixel, y_pixel):
        resolution = self.map_data["resolution"]
        origin_x, origin_y, _ = self.map_data["origin"]
        image_height = self.map_data["height"]
        x_world = x_pixel * resolution + origin_x
        y_world = (image_height - y_pixel) * resolution + origin_y
        return x_world, y_world

    def load_trajectory(self):
        points = []
        tmp = generate_trajectory()
        for x, y, theta in tmp:
            x_world, y_world = self.pixel_to_world(float(x), float(y))
            points.append((x_world, y_world, theta))
        self.get_logger().info(f"Trajectory points loaded: {len(points)} point(s).")
        return points

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        global current_pose_data, robot_path_data
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        siny_cosp = 2.0 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z)
        cosy_cosp = 1.0 - 2.0 * (msg.pose.pose.orientation.z ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        current_pose_data = {'x': x, 'y': y, 'yaw': yaw}
        if not robot_path_data or (abs(robot_path_data[-1]['x'] - x) > 0.05 or abs(robot_path_data[-1]['y'] - y) > 0.05):
            robot_path_data.append({'x': x, 'y': y})
        self.get_logger().info(f"현재 위치 업데이트: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

    def send_next_goal(self):
        global remaining_waypoints
        print(self.trajectory_points)
        if self.current_index >= len(self.trajectory_points):
            self.get_logger().info("✅ 모든 목표를 완료했습니다.")
            remaining_waypoints = 0
            # --- 추가: 모든 목표 도달 후 SLAM 맵 저장 및 전송 ---
            # self.save_and_transfer_map()
            return
        
        while self.latest_costmap is None:
            self.get_logger().info("🕒 costmap 대기 중...")
            rclpy.spin_once(self, timeout_sec=1)

        x, y, theta = self.trajectory_points[self.current_index]
        remaining_waypoints = len(self.trajectory_points) - self.current_index

        if self.check_goal_in_obstacle(x, y):
            self.get_logger().info("⚠️ 장애물 감지: 목표 취소")
            if not (x, y, theta) in self.revisited_paths:
                self.revisited_paths.add((x, y, theta))
                self.trajectory_points.append((x, y, theta))
                print("%f, %f, %f revisited paths append" % (x, y, theta))
            self.current_index += 1
            self.send_next_goal()
        else:
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.pose.position.x = x
            goal_msg.pose.pose.position.y = y
            goal_msg.pose.pose.orientation.w = 1.0  # 단순화를 위해 orientation은 고정
            goal_msg.pose.header.frame_id = "map"
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

            self.get_logger().info(f"🚀 목표 전송: x={x}, y={y}, theta={theta}")
            if not self._action_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("❌ Navigation2 서버에 연결할 수 없습니다.")
                return
            send_goal_future = self._action_client.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("❌ 목표가 거부되었습니다.")
            return
        self.get_logger().info("🚀 목표 수락됨, 진행 중...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        global remaining_waypoints
        result = future.result().result
        self.get_logger().info("✅ 목표 도달 완료!")
        self.current_index += 1
        remaining_waypoints = len(self.trajectory_points) - self.current_index
        # 만약 모든 목표를 완료했다면, 추가 명령 실행
        if self.current_index >= len(self.trajectory_points):
            self.get_logger().info("모든 목표 완료. SLAM 맵 저장 및 전송 시작합니다.")
            # self.save_and_transfer_map()
            return
        time.sleep(1)
        self.send_next_goal()

    # 추가: SLAM 맵 저장 및 scp 전송 함수
    def save_and_transfer_map(self):
        # SLAM 맵 저장 명령어 실행
        saver_cmd = "ros2 run nav2_map_server map_saver_cli -f final_map -t /slam/map"
        self.get_logger().info(f"맵 저장 명령 실행: {saver_cmd}")
        subprocess.run(saver_cmd, shell=True)
        # scp 명령어 실행 (경로와 사용자 이름을 알맞게 수정)
        scp_cmd = ("scp /home/storagy/Desktop/PINKLAB/src/storagy/final_map.pgm "
                   "/home/storagy/Desktop/PINKLAB/src/storagy/final_map.yaml "
                   "storagy@192.168.1.4:/home/사용자이름/Downloads/")
        self.get_logger().info(f"SCP 전송 명령 실행: {scp_cmd}")
        subprocess.run(scp_cmd, shell=True)

# 맵 이미지와 메타 데이터 로드 (Flask용)
map_image_path, map_data = load_map_image()

html_template = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>실시간 로봇 내비게이션</title>
    <style>
        #mapContainer { position: relative; display: inline-block; }
        #mapImage { display: block; }
        #overlay { position: absolute; top: 0; left: 0; cursor: crosshair; }
    </style>
</head>
<body>
    <h1>로봇 현재 위치 및 경로</h1>
    <div id="mapContainer">
        <img id="mapImage" src="/remap/get_map_image" alt="Map">
        <canvas id="overlay" width="{{ map_data.width }}" height="{{ map_data.height }}"></canvas>
    </div>
    <p id="coords">현재 로봇 위치:</p>
    <p id="waypoints_info">총 Waypoints: 0, 남은 Waypoints: 0</p>
    <button id="nextGoalBtn">재매핑 시작</button>
    <script>
        const resolution = {{ map_data.resolution }};
        const origin = {{ map_data.origin }};
        const mapWidth = {{ map_data.width }};
        const mapHeight = {{ map_data.height }};
        const overlay = document.getElementById('overlay');
        const ctx = overlay.getContext('2d');

        function mapToPixel(mapX, mapY) {
            let pixelX = (mapX - origin[0]) / resolution;
            let pixelY = mapHeight - ((mapY - origin[1]) / resolution);
            return { x: pixelX, y: pixelY };
        }

        document.getElementById('nextGoalBtn').addEventListener('click', function() {
            fetch('/remap/send_next_goal', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                alert("결과: " + JSON.stringify(data));
            })
            .catch(err => {
                console.error(err);
            });
        });

        async function updateOverlay() {
            try {
                // 로봇 경로 및 현재 위치 업데이트
                const poseResponse = await fetch('/remap/current_pose');
                const poseData = await poseResponse.json();
                const pathResponse = await fetch('/remap/robot_path');
                const pathData = await pathResponse.json();

                ctx.clearRect(0, 0, mapWidth, mapHeight);

                // trajectory waypoint 점들 (녹색 원)
                const trajResponse = await fetch('/remap/trajectory_points');
                const trajData = await trajResponse.json();
                if (trajData && trajData.length > 0) {
                    trajData.forEach(pt => {
                        const pixel = mapToPixel(pt[0], pt[1]);
                        ctx.beginPath();
                        ctx.arc(pixel.x, pixel.y, 3, 0, 2 * Math.PI);
                        ctx.fillStyle = 'green';
                        ctx.fill();
                    });
                }

                // 경로 그리기 (빨간 선)
                if (pathData && pathData.length > 0) {
                    ctx.beginPath();
                    pathData.forEach((pt, index) => {
                        const pixel = mapToPixel(pt.x, pt.y);
                        if (index === 0) {
                            ctx.moveTo(pixel.x, pixel.y);
                        } else {
                            ctx.lineTo(pixel.x, pixel.y);
                        }
                    });
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                // 현재 위치 그리기 (파란 원)
                if (poseData && poseData.x !== undefined) {
                    const pixel = mapToPixel(poseData.x, poseData.y);
                    ctx.beginPath();
                    ctx.arc(pixel.x, pixel.y, 5, 0, 2 * Math.PI);
                    ctx.fillStyle = 'blue';
                    ctx.fill();
                    document.getElementById('coords').textContent =
                        '현재 로봇 위치: x=' + poseData.x.toFixed(2) + ', y=' + poseData.y.toFixed(2);
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function updateWaypointsInfo() {
            try {
                const wpResponse = await fetch('/remap/waypoints');
                const wpData = await wpResponse.json();
                document.getElementById('waypoints_info').textContent =
                    "총 Waypoints: " + wpData.total_waypoints + ", 남은 Waypoints: " + wpData.remaining_waypoints;
            } catch (err) {
                console.error(err);
            }
        }

        setInterval(updateOverlay, 1000);
        setInterval(updateWaypointsInfo, 1000);
    </script>
</body>
</html>
'''

@remap_bp.route('/')
def index():
    return render_template_string(html_template, map_data=map_data)

@remap_bp.route('/get_map_image')
def get_map_image():
    global MAP_PNG_PATH
    if os.path.exists(MAP_PNG_PATH):
        return send_file(MAP_PNG_PATH, mimetype='image/png')
    else:
        return "Map image not found", 404

@remap_bp.route('/current_pose')
def current_pose():
    global current_pose_data
    return jsonify(current_pose_data if current_pose_data is not None else {})

@remap_bp.route('/robot_path')
def robot_path():
    global robot_path_data
    return jsonify(robot_path_data)

@remap_bp.route('/waypoints')
def waypoints():
    global total_waypoints, remaining_waypoints
    return jsonify({
        "total_waypoints": total_waypoints,
        "remaining_waypoints": remaining_waypoints
    })

@remap_bp.route('/trajectory_points')
def get_trajectory_points():
    global trajectory_points
    return jsonify(trajectory_points)

@remap_bp.route('/send_next_goal', methods=['POST'])
def send_next_goal_route():
    global total_waypoints, remaining_waypoints, trajectory_points
    if remap_node is None:
        return jsonify({'status': 'Node not ready'}), 500
    
    remap_node.current_index = 0
    trajectory_points = remap_node.load_trajectory()
    remap_node.send_next_goal()
    return jsonify({'status': 'Next goal triggered!'})

def init_remap():
    global remap_node
    remap_node = Nav2TrajectorySender()
    return remap_node