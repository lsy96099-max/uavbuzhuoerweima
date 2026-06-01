#!/usr/bin/env python3
# coding=utf-8
import rospy
import socket
import json
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32MultiArray

# ====================== 跨机器通信配置 ======================
BIND_IP = '192.168.31.70'
BIND_PORT = 9999
TEST_UAVS = [1, 2]

# ====================== 全局变量 ======================
qr_data = {}
for uav_id in TEST_UAVS:
    qr_data[uav_id] = {'poses': [], 'ids': []}

# ====================== 回调函数 ======================
def pose_callback(msg, uav_id):
    global qr_data
    poses = []
    for pose in msg.poses:
        poses.append({
            'x': round(pose.position.x, 4),
            'y': round(pose.position.y, 4),
            'z': round(pose.position.z, 4)
        })
    qr_data[uav_id]['poses'] = poses

def id_callback(msg, uav_id):
    global qr_data
    qr_data[uav_id]['ids'] = list(msg.data)

# ====================== 主函数（修改空数组处理） ======================
if __name__ == '__main__':
    rospy.init_node('ros_to_matlab_bridge')
    
    # 订阅话题
    for uav_id in TEST_UAVS:
        rospy.Subscriber(f'/qr_detection/uav{uav_id}/global_poses', PoseArray, pose_callback, callback_args=uav_id)
        rospy.Subscriber(f'/qr_detection/uav{uav_id}/ids', Int32MultiArray, id_callback, callback_args=uav_id)
        print(f"[Bridge] 已订阅无人机{uav_id}的二维码话题")
    
    # 创建TCP服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((BIND_IP, BIND_PORT))
    server_socket.listen(1)
    print(f"[Bridge] TCP服务器已启动：{BIND_IP}:{BIND_PORT}")
    print(f"[Bridge] 等待Matlab连接...")
    
    # 等待连接
    client_socket, client_addr = server_socket.accept()
    print(f"[Bridge] ✅ Matlab已连接：{client_addr}")
    
    rate = rospy.Rate(5)
    try:
        while not rospy.is_shutdown():
            # ✅ 核心修改：空数组替换为null，避免Matlab jsondecode bug
            send_data = {}
            for uav_id in TEST_UAVS:
                if len(qr_data[uav_id]['poses']) > 0:
                    send_data[str(uav_id)] = qr_data[uav_id]
                else:
                    send_data[str(uav_id)] = {'poses': None, 'ids': None}
            
            # 序列化数据
            data_json = json.dumps(send_data, separators=(',', ':'))
            data_bytes = data_json.encode('utf-8')
            data_len = len(data_bytes)
            
            # 发送数据
            client_socket.sendall(data_len.to_bytes(4, byteorder='little'))
            client_socket.sendall(data_bytes)
            
            print(f"[Bridge] 发送数据：长度={data_len}字节，内容={data_json}")
            
            rate.sleep()
    except ConnectionResetError:
        print(f"\n[Bridge] ❌ Matlab断开连接")
    except Exception as e:
        print(f"\n[Bridge] ❌ 异常：{e}")
    finally:
        try: client_socket.close()
        except: pass
        try: server_socket.close()
        except: pass
        print("[Bridge] 服务器已关闭")

