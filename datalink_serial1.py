#!/usr/bin/env python3
# coding=utf-8

import rospy

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import InertiaStamped
from nav_msgs.msg import Odometry

# ================= UAV =================
class UAV:

    def __init__(self, uav_id):

        self.id = uav_id

        # 初始坐标
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_z = 0.0

        self.init_ok = False

        # 当前坐标
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0

        # mission publisher
        self.mission_pub = rospy.Publisher(
            f"/fcu_bridge/mission_{uav_id:03d}",
            InertiaStamped,
            queue_size=10
        )

        # MATLAB 轨迹
        rospy.Subscriber(
            f"/uav{uav_id}/cmd_pose",
            PoseStamped,
            self.cmd_callback
        )

        # odom
        rospy.Subscriber(
            f"/odom_global_{uav_id:03d}",
            Odometry,
            self.odom_callback
        )

        print(f"[UAV{uav_id}] init ok")

    # ================= ODOM =================
    def odom_callback(self, msg):

        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.pos_z = msg.pose.pose.position.z

        # 记录初始点
        if not self.init_ok:

            self.init_x = self.pos_x
            self.init_y = self.pos_y
            self.init_z = self.pos_z

            self.init_ok = True

            print(f"[UAV{self.id}] 初始坐标:")
            print(self.init_x, self.init_y, self.init_z)

    # ================= MATLAB =================
    def cmd_callback(self, msg):

        if not self.init_ok:
            print(f"[UAV{self.id}] 等待定位...")
            return

        dx = msg.pose.position.x
        dy = msg.pose.position.y
        # 核心修改：Z轴直接用MATLAB的dz作为绝对坐标（不再加init_z）
        # 如果你需要MATLAB的dz是“相对初始z的增量”，则保留pz = self.init_z + dz，否则直接pz = dz
        pz = msg.pose.position.z  # Z轴绝对坐标，由MATLAB直接指定
        px = self.init_x + dx     # XY保持相对初始值的逻辑
        py = self.init_y + dy

        print(f"\n[UAV{self.id}] MATLAB指令:")
        print(f"XY相对值(dx,dy): {dx}, {dy} | Z绝对值: {pz}")

        print(f"[UAV{self.id}] 转换后目标(全局):")
        print(px, py, pz)

        # ================= mission =================
        mission = InertiaStamped()

        # position
        mission.inertia.com.x = px
        mission.inertia.com.y = py
        # 符号适配：飞控NED坐标系Z+向下，所以ENU的Z向上需要传-NED_Z
        # 这里pz是无人机ENU坐标系的绝对高度（向上为正），所以转NED要取负
        mission.inertia.com.z = -pz  

        # 其他字段保持不变
        mission.inertia.ixx = 0
        mission.inertia.ixy = 0
        mission.inertia.ixz = 0
        mission.inertia.iyy = 0
        mission.inertia.iyz = 0
        mission.inertia.izz = 0
        mission.inertia.m = 0

        self.mission_pub.publish(mission)
        print(f"[UAV{self.id}] mission已发布 | NED Z: {-pz}")


# ================= MAIN =================
class MultiUAV:

    def __init__(self):

        rospy.init_node("multi_uav_datalink")

        self.uavs = []

        for i in range(1,5):  # 初始化UAV1~UAV4
            self.uavs.append(UAV(i))

        print("\n========================")
        print("Multi UAV Datalink Start")
        print("========================")

    def spin(self):
        rospy.spin()

# ================= MAIN =================
if __name__ == "__main__":

    node = MultiUAV()

    node.spin()

