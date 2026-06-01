#!/usr/bin/env python3
# coding=utf-8
import torch
import cv2
import numpy as np
import subprocess
import multiprocessing as mp
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Int32MultiArray

# ====================== 四台无人机的固定配置 ======================
UAV_CONFIGS = [
    {"uav_id": 1, "camera_url": "tcp://192.168.31.34:2024"},
    {"uav_id": 2, "camera_url": "tcp://192.168.31.178:2024"},
    {"uav_id": 3, "camera_url": "tcp://192.168.31.8:2024"},
    {"uav_id": 4, "camera_url": "tcp://192.168.31.54:2024"},
]

# ====================== 无人机状态类 ======================
class UAVState:
    def __init__(self):
        self.exec_state = 0  # WAITING=0, TRACKING=1, LOST=2
        self.is_detected = False
        self.num_count_vision_lost = 0
        self.num_count_vision_regain = 0
        self.VISION_THRES = 8
        self.g_kp_detect = [0.5, 0.8, 0.8]
        self.Detection_distance = 3.0
        self.iou_threshold = 0.6
        self.drone_global_x = 0.0
        self.drone_global_y = 0.0
        self.drone_global_z = 0.0
        self.drone_global_yaw = 0.0

# ====================== 原有函数 ======================
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

# ====================== 视频显示函数 ======================
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
    window_name = f"UAV{uav_id} Downward View (Tracking)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while not rospy.is_shutdown():
        raw_frame = process.stdout.read(W_img * H_img * 3)
        if len(raw_frame) != W_img * H_img * 3:
            rospy.logwarn(f"[UAV{uav_id}] 视频流中断，尝试重连...")
            break
        frame = np.frombuffer(raw_frame, np.uint8).reshape((H_img, W_img, 3))
        with lock:
            np.copyto(np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape(H_img, W_img, 3), frame)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)
    process.terminate()

# ====================== ✅ 修复：odom回调函数参数解包 ======================
def odom_global_callback(msg, args):
    # 关键修复：先解包元组参数
    uav_state, uav_id = args
    
    uav_state.drone_global_x = msg.pose.pose.position.x
    uav_state.drone_global_y = msg.pose.pose.position.y
    uav_state.drone_global_z = msg.pose.pose.position.z
    
    if not hasattr(odom_global_callback, f'printed_{uav_id}'):
        print(f"[UAV{uav_id}] 成功收到odom数据！初始坐标: ({uav_state.drone_global_x:.2f}, {uav_state.drone_global_y:.2f}, 高度: {uav_state.drone_global_z:.2f}m)")
        setattr(odom_global_callback, f'printed_{uav_id}', True)
    
    qw = msg.pose.pose.orientation.w
    qx = msg.pose.pose.orientation.x
    qy = msg.pose.pose.orientation.y
    qz = msg.pose.pose.orientation.z
    uav_state.drone_global_yaw = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))

# ====================== 二维码检测函数 ======================
def qr_code_detection(uav_id, uav_state, shared_array, lock):
    # ---------------------- 下视相机核心参数 ----------------------
    W_img = 640
    H_img = 480
    FOV_x = 90.0   # 你的下视相机水平视场角
    FOV_y = 70.0   # 你的下视相机垂直视场角
    W_real = 0.15  # 二维码实际物理宽度（米，必须准确！）
    k = 1.0        # 比例系数，可根据实际测量微调
    
    # 计算相机焦距（像素）
    f_x = (W_img / 2) / np.tan(np.radians(FOV_x / 2))
    f_y = (H_img / 2) / np.tan(np.radians(FOV_y / 2))
    print(f"[UAV{uav_id}] 下视相机内参：f_x={f_x:.4f}px, f_y={f_y:.4f}px")
    print(f"[UAV{uav_id}] 二维码实际宽度：{W_real}m")
    
    # YOLO模型
    model = torch.hub.load('/home/lvshunyao/桌面/yolov5', 'custom', path='/home/lvshunyao/桌面/best.pt', source='local')
    model.conf = 0.7
    model.iou = 0.4  # 降低NMS阈值，避免相近二维码被过滤
    
    target_locked = False
    target_box = None
    detection_window_name = f"UAV{uav_id} QR Detection (Downward)"
    cv2.namedWindow(detection_window_name, cv2.WINDOW_NORMAL)
    
    # 发布话题
    qr_poses_pub = rospy.Publisher(f'/qr_detection/uav{uav_id}/global_poses', PoseArray, queue_size=10)
    qr_ids_pub = rospy.Publisher(f'/qr_detection/uav{uav_id}/ids', Int32MultiArray, queue_size=10)
    
    while not rospy.is_shutdown():
        with lock:
            frame = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape(H_img, W_img, 3).copy()
        results = model(frame)
        detections = results.pred[0]
        detections = detections[detections[:, 4] >= 0.7]
        
        # 多二维码画框（不同颜色+编号，避免重叠）
        all_qr_data = []  # 存储所有二维码的坐标数据
        for idx, (*xyxy, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, xyxy)
            qr_id = idx + 1  # 二维码编号从1开始
            
            # 不同颜色区分：跟踪目标用红色粗框，其他用蓝色细框
            if target_locked and target_box is not None and calculate_iou(target_box, xyxy) >= uav_state.iou_threshold:
                color = (0, 0, 255)  # 红色=跟踪目标
                thickness = 3
                label = f'Tracking QR{qr_id}: {conf:.2f}'
            else:
                color = (255, 0, 0)  # 蓝色=普通检测
                thickness = 2
                label = f'QR{qr_id}: {conf:.2f}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # 标签位置错开，避免重叠
            cv2.putText(frame, label, (x1, y1 - 10 - idx*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 计算该二维码的相对坐标
            W_qr = x2 - x1
            cx_qr = (x1 + x2) / 2
            cy_qr = (y1 + y2) / 2
            cx_img = W_img / 2
            cy_img = H_img / 2
            
            dz_rel = k * (W_real * f_x) / W_qr
            dx_pix = cx_qr - cx_img
            dy_pix = cy_qr - cy_img
            dx_rel = (dx_pix / f_x) * dz_rel
            dy_rel = (dy_pix / f_y) * dz_rel
            
            # 存储所有二维码数据
            all_qr_data.append({
                'id': qr_id,
                'dx': dx_rel,
                'dy': dy_rel,
                'dz': dz_rel,
                'box': xyxy
            })
        
        # 单目标跟踪逻辑（保留，只跟踪第一个锁定的目标）
        if uav_state.exec_state == 0:
            if len(detections) > 0:
                target_box = detections[0][:4].tolist()
                target_locked = True
                uav_state.is_detected = True
                uav_state.exec_state = 1
                print(f"[UAV{uav_id}] 下视追踪：锁定目标QR1")
            else:
                if not hasattr(qr_code_detection, f'last_wait_print_{uav_id}') or rospy.Time.now().to_sec() - getattr(qr_code_detection, f'last_wait_print_{uav_id}') > 30:
                    print(f"[UAV{uav_id}] 下视追踪：等待目标二维码...")
                    setattr(qr_code_detection, f'last_wait_print_{uav_id}', rospy.Time.now().to_sec())
        
        elif uav_state.exec_state == 1:
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
                        uav_state.exec_state = 2
                        print(f"[UAV{uav_id}] 下视追踪：丢失目标")
            else:
                uav_state.num_count_vision_lost += 1
                if uav_state.num_count_vision_lost > uav_state.VISION_THRES:
                    uav_state.exec_state = 2
                    print(f"[UAV{uav_id}] 下视追踪：丢失目标")
        
        elif uav_state.exec_state == 2:
            if len(detections) > 0:
                target_box = detections[0][:4].tolist()
                uav_state.exec_state = 1
                print(f"[UAV{uav_id}] 下视追踪：重新锁定目标QR1")
        
        # 发布所有二维码的坐标
        qr_global_poses = PoseArray()
        qr_global_poses.header.stamp = rospy.Time.now()
        qr_global_poses.header.frame_id = "map"
        qr_ids = Int32MultiArray()
        
        for qr in all_qr_data:
            dx_rel = qr['dx']
            dy_rel = qr['dy']
            dz_rel = qr['dz']
            
            # 转换为全局坐标
            cos_yaw = np.cos(uav_state.drone_global_yaw)
            sin_yaw = np.sin(uav_state.drone_global_yaw)
            qr_global_x = uav_state.drone_global_x + dx_rel * cos_yaw - dy_rel * sin_yaw
            qr_global_y = uav_state.drone_global_y + dx_rel * sin_yaw + dy_rel * cos_yaw
            qr_global_z = uav_state.drone_global_z - dz_rel
            
            # 构造消息
            qr_pose = Pose()
            qr_pose.position.x = qr_global_x
            qr_pose.position.y = qr_global_y
            qr_pose.position.z = qr_global_z
            qr_pose.orientation.w = 1.0
            
            qr_global_poses.poses.append(qr_pose)
            qr_ids.data.append(qr['id'])
        
        qr_poses_pub.publish(qr_global_poses)
        qr_ids_pub.publish(qr_ids)
        
        # 打印所有二维码的坐标（每2秒一次，避免刷屏）
        if len(detections) > 0 and rospy.Time.now().to_sec() % 2 < 0.1:
            print(f"\n[UAV{uav_id}] ======================================")
            print(f"检测到 {len(detections)} 个二维码：")
            for qr in all_qr_data:
                print(f"  QR{qr['id']}: dx={qr['dx']:.2f}m, dy={qr['dy']:.2f}m, 高度差={qr['dz']:.2f}m")
            print("======================================")
        
        cv2.imshow(detection_window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow(detection_window_name)

# ====================== 单台无人机运行函数 ======================
def run_single_uav(uav_id, camera_url):
    rospy.init_node(f'qr_detection_downward_{uav_id:03d}', anonymous=False)
    uav_state = UAVState()
    
    odom_topic = f'/odom_global_{uav_id:03d}'
    # ✅ 这里保持不变，参数会被打包成元组传递给回调函数
    rospy.Subscriber(odom_topic, Odometry, odom_global_callback, callback_args=(uav_state, uav_id))
    print(f"[UAV{uav_id}] 订阅odom话题: {odom_topic}")
    print(f"[UAV{uav_id}] 下视相机流地址: {camera_url}")
    
    W_img = 640
    H_img = 480
    shared_array = mp.Array('B', W_img * H_img * 3)
    lock = mp.Lock()
    
    p_display = mp.Process(target=video_display, args=(uav_id, camera_url, shared_array, lock))
    p_display.start()
    
    qr_code_detection(uav_id, uav_state, shared_array, lock)
    
    p_display.join()

# ====================== 主函数 ======================
if __name__ == '__main__':
    uav_processes = []
    for config in UAV_CONFIGS:
        uav_id = config["uav_id"]
        camera_url = config["camera_url"]
        p = mp.Process(target=run_single_uav, args=(uav_id, camera_url))
        p.start()
        uav_processes.append(p)
        print(f"[主进程] 已启动UAV{uav_id}的下视追踪程序")
    
    for p in uav_processes:
        p.join()

