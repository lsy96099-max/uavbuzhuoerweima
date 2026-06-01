#!/usr/bin/env python3
# coding=utf-8
import torch
import cv2
import numpy as np
import subprocess
import multiprocessing as mp
import threading
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Int32MultiArray

# ====================== 四台无人机的固定配置（核心：这里写死你的4台IP/ID） ======================
UAV_CONFIGS = [
    {"uav_id": 1, "camera_url": "tcp://192.168.31.34:2024"},  # 无人机1的固定IP+端口
    {"uav_id": 2, "camera_url": "tcp://192.168.31.178:2024"},  # 无人机2的固定IP+端口
    {"uav_id": 3, "camera_url": "tcp://192.168.31.8:2024"},  # 无人机3的固定IP+端口
    {"uav_id": 4, "camera_url": "tcp://192.168.31.54:2024"},  # 无人机4的固定IP+端口
]

# ====================== 原有全局变量（改为类封装，避免多进程冲突） ======================
class UAVState:
    def __init__(self):
        self.exec_state = 0  # WAITING=0, TRACKING=1, LOST=2
        self.is_detected = False
        self.num_count_vision_lost = 0
        self.num_count_vision_regain = 0
        self.VISION_THRES = 5
        self.g_kp_detect = [0.5, 0.8, 0.8]
        self.Detection_distance = 1.5
        self.iou_threshold = 0.5
        self.drone_global_x = 0.0
        self.drone_global_y = 0.0
        self.drone_global_z = 0.0
        self.drone_global_yaw = 0.0

# ====================== 原有函数（完全保留） ======================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

# ====================== 视频显示函数（适配单台无人机配置） ======================
def video_display(uav_id, camera_url, shared_array, lock):
    process = subprocess.Popen([
        'ffmpeg',
        '-i', camera_url,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-'
    ], stdout=subprocess.PIPE)
    W_img = 640
    H_img = 480
    window_name = f"UAV{uav_id} Real-Time Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while not rospy.is_shutdown():
        raw_frame = process.stdout.read(W_img * H_img * 3)
        if len(raw_frame) != W_img * H_img * 3:
            break
        frame = np.frombuffer(raw_frame, np.uint8).reshape((H_img, W_img, 3))
        with lock:
            np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape(H_img, W_img, 3), frame)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)
    process.terminate()

# ====================== odom回调（适配单台无人机状态） ======================
def odom_global_callback(msg, uav_state, uav_id):
    uav_state.drone_global_x = msg.pose.pose.position.x
    uav_state.drone_global_y = msg.pose.pose.position.y
    uav_state.drone_global_z = msg.pose.pose.position.z
    
    # 只打印一次初始值
    if not hasattr(odom_global_callback, f'printed_{uav_id}'):
        print(f"[UAV{uav_id}] 成功收到odom数据！初始坐标: ({uav_state.drone_global_x:.2f}, {uav_state.drone_global_y:.2f}, {uav_state.drone_global_z:.2f})")
        setattr(odom_global_callback, f'printed_{uav_id}', True)
    
    qw = msg.pose.pose.orientation.w
    qx = msg.pose.pose.orientation.x
    qy = msg.pose.pose.orientation.y
    qz = msg.pose.pose.orientation.z
    uav_state.drone_global_yaw = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))

# ====================== 二维码检测函数（适配单台无人机配置） ======================
def qr_code_detection(uav_id, uav_state, shared_array, lock):
    W_img = 640
    H_img = 480
    FOV_x = 53.5
    FOV_y = 41.4
    W_real = 0.2
    k = 1.1
    
    f_x = (W_img / 2) / np.tan(np.radians(FOV_x / 2))
    f_y = (H_img / 2) / np.tan(np.radians(FOV_y / 2))
    print(f"[UAV{uav_id}] f_x: {f_x:.4f}, f_y: {f_y:.4f} px")
    
    # YOLO模型路径（所有无人机共用同一个模型文件）
    model = torch.hub.load('/home/lvshunyao/桌面/yolov5', 'custom', path='/home/lvshunyao/桌面/best.pt', source='local')
    
    target_locked = False
    target_box = None
    detection_window_name = f"UAV{uav_id} QR Detection"
    cv2.namedWindow(detection_window_name, cv2.WINDOW_NORMAL)
    
    # 参数化发布话题
    qr_poses_pub = rospy.Publisher(f'/qr_detection/uav{uav_id}/global_poses', PoseArray, queue_size=10)
    qr_ids_pub = rospy.Publisher(f'/qr_detection/uav{uav_id}/ids', Int32MultiArray, queue_size=10)
    
    while not rospy.is_shutdown():
        with lock:
            frame = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape(H_img, W_img, 3).copy()
        results = model(frame)
        detections = results.pred[0]
        conf_threshold = 0.6
        detections = detections[detections[:, 4] >= conf_threshold]
        
        # 多二维码画框逻辑
        for idx, (*xyxy, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, xyxy)
            if target_locked and target_box is not None and calculate_iou(target_box, xyxy) >= uav_state.iou_threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f'Tracking QR{idx+1}: {conf:.2f}'
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'QR{idx+1}: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 单目标跟踪逻辑（修正缩进，纳入while循环）
        if uav_state.exec_state == 0:  # WAITING
            if len(detections) > 0:
                target_box = detections[0][:4].tolist()
                target_locked = True
                uav_state.is_detected = True
                uav_state.exec_state = 1  # TRACKING
                print(f"[UAV{uav_id}] Target locked")
            else:
                # 每30秒打印一次，避免刷屏
                if not hasattr(qr_code_detection, f'last_wait_print_{uav_id}') or rospy.Time.now().to_sec() - getattr(qr_code_detection, f'last_wait_print_{uav_id}') > 30:
                    print(f"[UAV{uav_id}] Waiting for target")
                    setattr(qr_code_detection, f'last_wait_print_{uav_id}', rospy.Time.now().to_sec())
        
        elif uav_state.exec_state == 1:  # TRACKING
            if len(detections) > 0:
                found = False
                for *xyxy, conf, cls in detections:
                    iou = calculate_iou(target_box, xyxy)
                    if iou >= uav_state.iou_threshold:
                        target_box = xyxy
                        found = True
                        uav_state.num_count_vision_regain += 1
                        uav_state.num_count_vision_lost = 0
                        break
                if not found:
                    uav_state.num_count_vision_lost += 1
                    if uav_state.num_count_vision_lost > uav_state.VISION_THRES:
                        uav_state.exec_state = 2  # LOST
                        print(f"[UAV{uav_id}] Target lost")
                else:
                    uav_state.num_count_vision_lost = 0
                    x1, y1, x2, y2 = map(int, target_box)
                    W_qr = x2 - x1
                    cx_qr = (x1 + x2) / 2
                    cy_qr = (y1 + y2) / 2
                    cx_img = W_img / 2
                    cy_img = H_img / 2
                    
                    dx = cx_qr - cx_img
                    dy = cy_qr - cy_img
                    dz_m = k * (W_real * f_x) / W_qr
                    dx_m = k * (dx / f_x) * dz_m
                    dy_m = k * (dy / f_y) * dz_m
                    
                    label = f'dx: {dx_m:.2f}m, dy: {dy_m:.2f}m'
                    cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            else:
                uav_state.num_count_vision_lost += 1
                if uav_state.num_count_vision_lost > uav_state.VISION_THRES:
                    uav_state.exec_state = 2  # LOST
                    print(f"[UAV{uav_id}] Target lost")
        
        elif uav_state.exec_state == 2:  # LOST
            if len(detections) > 0:
                target_box = detections[0][:4].tolist()
                uav_state.exec_state = 1  # TRACKING
                print(f"[UAV{uav_id}] Target regained")
        
        # 解算所有二维码全局坐标并发布
        qr_global_poses = PoseArray()
        qr_global_poses.header.stamp = rospy.Time.now()
        qr_global_poses.header.frame_id = "map"
        qr_ids = Int32MultiArray()
        
        for idx, (*xyxy, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, xyxy)
            W_qr = x2 - x1
            cx_qr = (x1 + x2) / 2
            cy_qr = (y1 + y2) / 2
            cx_img = W_img / 2
            cy_img = H_img / 2
            
            # 坐标转换
            dx_cam = cx_qr - cx_img
            dy_cam = cy_qr - cy_img
            dz_cam = k * (W_real * f_x) / W_qr
            dx_m_cam = k * (dx_cam / f_x) * dz_cam
            dy_m_cam = k * (dy_cam / f_y) * dz_cam
            
            dx_body = dz_cam
            dy_body = dx_m_cam
            dz_body = -dy_m_cam
            
            cos_yaw = np.cos(uav_state.drone_global_yaw)
            sin_yaw = np.sin(uav_state.drone_global_yaw)
            qr_global_x = uav_state.drone_global_x + dx_body * cos_yaw - dy_body * sin_yaw
            qr_global_y = uav_state.drone_global_y + dx_body * sin_yaw + dy_body * cos_yaw
            qr_global_z = uav_state.drone_global_z + dz_body
            
            qr_pose = Pose()
            qr_pose.position.x = qr_global_x
            qr_pose.position.y = qr_global_y
            qr_pose.position.z = qr_global_z
            qr_pose.orientation.w = 1.0
            
            qr_global_poses.poses.append(qr_pose)
            qr_ids.data.append(idx+1)
        
        qr_poses_pub.publish(qr_global_poses)
        qr_ids_pub.publish(qr_ids)
        
        # 打印调试信息
        if len(detections) > 0 and rospy.Time.now().to_sec() % 5 < 0.1:
            print(f"\n[UAV{uav_id}] 检测到{len(detections)}个二维码")
            print(f"无人机坐标: ({uav_state.drone_global_x:.2f}, {uav_state.drone_global_y:.2f}, {uav_state.drone_global_z:.2f}) m")
        
        cv2.imshow(detection_window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow(detection_window_name)

# ====================== 单台无人机的核心运行函数 ======================
def run_single_uav(uav_id, camera_url):
    # 初始化ROS节点（每个进程独立节点）
    rospy.init_node(f'qr_detection_{uav_id:03d}', anonymous=False)
    
    # 初始化该无人机的状态
    uav_state = UAVState()
    
    # 订阅odom话题
    odom_topic = f'/odom_global_{uav_id:03d}'
    rospy.Subscriber(odom_topic, Odometry, odom_global_callback, callback_args=(uav_state, uav_id))
    print(f"[UAV{uav_id}] 订阅odom话题: {odom_topic}")
    print(f"[UAV{uav_id}] 相机流地址: {camera_url}")
    
    # 初始化共享内存和锁
    W_img = 640
    H_img = 480
    shared_array = mp.Array('B', W_img * H_img * 3)
    lock = mp.Lock()
    
    # 启动视频显示进程
    p_display = mp.Process(target=video_display, args=(uav_id, camera_url, shared_array, lock))
    p_display.start()
    
    # 启动二维码检测（当前进程运行，方便ROS回调）
    qr_code_detection(uav_id, uav_state, shared_array, lock)
    
    # 等待视频进程结束
    p_display.join()

# ====================== 主函数：批量启动四台无人机 ======================
if __name__ == '__main__':
    # 存储所有无人机进程
    uav_processes = []
    
    # 遍历四台无人机配置，启动进程
    for config in UAV_CONFIGS:
        uav_id = config["uav_id"]
        camera_url = config["camera_url"]
        
        # 启动单台无人机的进程
        p = mp.Process(target=run_single_uav, args=(uav_id, camera_url))
        p.start()
        uav_processes.append(p)
        print(f"[主进程] 已启动UAV{uav_id}的检测程序")
    
    # 等待所有进程结束
    for p in uav_processes:
        p.join()

