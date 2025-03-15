import os
import time
import math
import yaml
import numpy as np
import cv2
import heapq
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

from utils.config import YAML_PATH, MAP_PNG_PATH, DEBUG_IMG_DIR

os.makedirs(DEBUG_IMG_DIR, exist_ok=True)

# =========================
# 전역 변수 및 설정
# =========================
current_pose_data = None  # 로봇의 현재 위치 저장
TRAJECTORY_FILE = "static/trajectory.txt"
node_amcl = None

# =========================
# YAML 설정 로드
# =========================
def load_map_yaml():
    if not os.path.exists(YAML_PATH):
        print("❌ YAML 파일을 찾을 수 없습니다.")
        return None
    with open(YAML_PATH, 'r') as file:
        return yaml.safe_load(file)

# =========================
# 월드 좌표 → 픽셀 좌표 변환
# =========================
def world_to_pixel(x_world, y_world, map_config, image_height):
    resolution = map_config["resolution"]
    origin_x, origin_y, _ = map_config["origin"]
    x_pixel = int((x_world - origin_x) / resolution)
    y_pixel = int(image_height - ((y_world - origin_y) / resolution))
    return x_pixel, y_pixel

# =========================
# ROS2 노드: 현재 위치 업데이트
# =========================
class AmclPoseSubscriber(Node):
    def __init__(self):
        super().__init__('amcl_pose_subscriber')
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10)
    
    def pose_callback(self, msg):
        global current_pose_data
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        siny_cosp = 2.0 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z)
        cosy_cosp = 1.0 - 2.0 * (msg.pose.pose.orientation.z ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        current_pose_data = {'x': x, 'y': y, 'yaw': yaw}
        self.get_logger().info(f"현재 위치 업데이트: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

# =========================
# A* 경로 탐색 함수
# =========================
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def grid_a_star_search(start, goal, node_set, step):
    directions = [
        (step, 0), (-step, 0), (0, step), (0, -step),
        (step, step), (step, -step), (-step, step), (-step, -step)
    ]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in node_set:
                move_cost = np.hypot(dx, dy) / step
                new_cost = cost_so_far[current] + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
    if goal not in came_from:
        return []
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = came_from[cur]
    path.append(start)
    path.reverse()
    return path

# =========================
# A* 경로 따라가기 함수
# =========================
def move_along_astar_path(start_pos, path):
    traj = []
    robot = np.array([start_pos[0], start_pos[1], 0.0])
    for point in path:
        robot[:2] = np.array(point)
        traj.append(robot.copy())
    return traj

# =========================
# 경로 스무딩 함수
# =========================
def smooth_path(path, window_size=3):
    if len(path) < window_size:
        return path
    smoothed = []
    for i in range(len(path)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(path), i + window_size + 1)
        pts = np.array(path[start_idx:end_idx])
        avg = np.mean(pts, axis=0)
        smoothed.append((avg[0], avg[1], avg[2]))
    return smoothed

# =========================
# 메인 로직: 순서 재배치 (웨이포인트 먼저 생성 → 로봇 위치 수신 → 경로 생성)
# =========================
def generate_trajectory():
    global node_amcl, current_pose_data

    # (1) 맵 이미지에서 웨이포인트(grid_nodes) 생성
    print("🔄 (1) 맵 이미지에서 웨이포인트 생성 중...")
    map_config = load_map_yaml()
    if map_config is None:
        print("❌ map.yaml 로드 실패")
        msg = "map.yaml 파일을 찾을 수 없습니다"
        return (msg, [])

    image = cv2.imread(MAP_PNG_PATH, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ 파일 {MAP_PNG_PATH}를 찾을 수 없습니다.")
        msg = "map.png 파일을 찾을 수 없습니다"
        return (msg, [])
    h, w = image.shape

    # 이진화: 검정(벽)이 255, 이동 가능 영역이 0
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # (2) 내부 이동 가능한 영역 찾기 (흰색과 검은색 반전)
    start_x, start_y = 128, 128
    if binary[start_y, start_x] != 0:
        print(":x: 시작 지점이 이동 불가능한 위치입니다. 다시 설정하세요.")
        msg = "start_x, start_y를 다시 설정해주세요"
        return (msg, [])
    flood_filled = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask, (start_x, start_y), 255)
    interior = cv2.bitwise_and(flood_filled, cv2.bitwise_not(binary))
    cv2.imwrite(os.path.join(DEBUG_IMG_DIR, "interior.png"), interior)
    print(f":흰색_확인_표시: interior 이미지가 {DEBUG_IMG_DIR} 에 저장되었습니다.")

    pixel_to_meter = map_config["resolution"]
    margin_m = 0.4
    margin_pixels = margin_m / pixel_to_meter
    dist_transform = cv2.distanceTransform(interior, cv2.DIST_L2, 5)

    grid_size = 30
    grid_nodes = []
    node_set = set()
    for yy in range(grid_size//2, h, grid_size):
        for xx in range(grid_size//2, w, grid_size):
            if interior[yy, xx] == 255 and dist_transform[yy, xx] >= margin_pixels:
                grid_nodes.append((xx, yy))
                node_set.add((xx, yy))
    if not grid_nodes:
        print(":x: 웨이포인트가 없습니다. 이미지 처리를 확인하세요.")
        msg = "웨이포인트가 없습니다. 평면도를 확인해주세요"
        return (msg, [])
    print(f"✅ 웨이포인트 생성 완료: 총 {len(grid_nodes)}개")

    # (2-1) 웨이포인트 미리보기 저장 (옵션)
    partial_result = cv2.cvtColor(interior, cv2.COLOR_GRAY2BGR)
    for node in grid_nodes:
        cv2.circle(partial_result, node, 3, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(DEBUG_IMG_DIR, "waypoints_preview.png"), partial_result)
    print("🔎 waypoint 미리보기 이미지(waypoints_preview.png) 저장 완료")

    # (3) ROS2 초기화 및 로봇 현재 위치 수신
    print("🔄 (2) 로봇 현재 위치 수신 대기...")

    # node_amcl = AmclPoseSubscriber()
    wait_time = 0
    while current_pose_data is None and wait_time < 30:
        rclpy.spin_once(node_amcl, timeout_sec=1.0)
        wait_time += 1
    if current_pose_data is None:
        print("❌ AMCL에서 위치를 받지 못했습니다. 종료합니다.")
        msg = "로봇의 현재 위치를 설정해주세요"
        return (msg, [])
    x_world, y_world = current_pose_data['x'], current_pose_data['y']
    print(f"🌍 로봇 위치 수신 완료: x={x_world:.2f}, y={y_world:.2f}")

    # (4) 로봇 위치를 픽셀 좌표로 변환 후, 가장 가까운 웨이포인트 선택 (시작점)
    sx, sy = world_to_pixel(x_world, y_world, map_config, h)
    if binary[sy, sx] != 0:
        print("❌ 로봇 위치가 이동 불가능한 영역(또는 벽)입니다. 종료합니다.")
        msg = "로봇 위치가 이동 불가능한 영역입니다."
        return (msg, [])
    # 현재 로봇 위치를 기존 웨이포인트 중 가장 가까운 것으로 강제 스냅
    start = min(grid_nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array((sx, sy))))
    robot_position = start
    print(f"✅ 로봇과 가장 가까운 웨이포인트: {start}")

    # (5) 전체 시뮬레이션 루프 (중복 제거 적용)
    visited_nodes = set()
    robot_position = start
    visited_nodes.add(robot_position)
    simulation_steps = 0
    max_iterations = 1000
    trajectory_sim = []
    trajectory_set = set()

    while len(visited_nodes) < len(grid_nodes) and simulation_steps < max_iterations:
        unvisited = [node for node in grid_nodes if node not in visited_nodes]
        if not unvisited:
            break
        next_target = min(unvisited, key=lambda n: np.linalg.norm(np.array(robot_position) - np.array(n)))
        path = grid_a_star_search(robot_position, next_target, node_set, grid_size)
        if not path:
            visited_nodes.add(next_target)
            robot_position = next_target
            simulation_steps += 1
            continue
        astar_traj = move_along_astar_path(robot_position, path)
        # smoothed_traj = smooth_path(astar_traj, window_size=3)
        for pt in astar_traj:
            pt_tuple = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
            if pt_tuple not in trajectory_set:
                trajectory_sim.append(pt)
                trajectory_set.add(pt_tuple)
        robot_position = tuple(astar_traj[-1][:2])
        visited_nodes.add(next_target)
        simulation_steps += 1

    trajectory = []
    # (6) 경로 저장 (픽셀 좌표를 소수점 둘째자리 float 형식으로)
    with open(TRAJECTORY_FILE, "w") as f:
        for idx, pt in enumerate(trajectory_sim):
            trajectory.append((pt[0], pt[1], 0.00))
            f.write(f"{idx}: x={pt[0]:.2f}, y={pt[1]:.2f}, theta=0.00\n")
    print(f"✅ 최종 경로 저장 완료: {len(trajectory_sim)} 포인트 생성.")

    # (7) 최종 결과 시각화 및 저장
    final_result = cv2.cvtColor(interior, cv2.COLOR_GRAY2BGR)
    for node in grid_nodes:
        cv2.circle(final_result, node, 3, (255, 0, 0), -1)
    if len(trajectory_sim) > 1:
        traj_points = np.array([[int(pt[0]), int(pt[1])] for pt in trajectory_sim])
        cv2.polylines(final_result, [traj_points], False, (0, 255, 255), thickness=2)
    # 로봇의 최종 위치는 trajectory_sim의 마지막 좌표 사용
    if trajectory_sim:
        last_pos = trajectory_sim[-1]
        cv2.circle(final_result, (int(last_pos[0]), int(last_pos[1])), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(DEBUG_IMG_DIR, "final_astar_mapping.png"), final_result)
    print("✅ 최종 A* 기반 경로 시각화 이미지(final_astar_mapping.png) 저장 완료")
    msg = "경로 생성에 성공했습니다"
    return (msg, trajectory)

def init_pose_sub():
    global node_amcl
    node_amcl = AmclPoseSubscriber()
    return

if __name__ == "__main__":
    trajectory = generate_trajectory()
    print(trajectory)
