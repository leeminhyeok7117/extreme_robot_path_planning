import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import os, sys
import time

import numpy as np
from math import *

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, Point, PointStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry

from tf_transformations import euler_from_quaternion

ANGULAR_THRESHOLD = 20
ANGULAR_THRESHOLD_STOP = 3
ANGULAR_THRESHOLD_ODOM = 2
DISTANCE_LIDAR_BASE = 0.167
FORWARD_SPEED = 5.0
INDEX_THRESHOLD = 0.2
Kp = 0.3

class FollowPoint(Node):
    def __init__(self):
        super().__init__('follow_point')
        qos_profile_1 = QoSProfile(depth = 1)
        self.erp_pub = self.create_publisher(Twist, '/erp_pub', qos_profile_1)
        
        self.goal_sub = self.create_subscription(PointCloud2, '/goal_point', self.goal_callback, qos_profile_1)
        self.heading_sub = self.create_subscription(Float32, '/heading', self.heading_callback, qos_profile_1)
        self.odom_sub = self.create_subscription(Odometry, '/Odometry', self.odom_callback, qos_profile_1)
        self.last_odom_time = 0.0
        
        self.create_timer(0.1, self.pub_what)
        
        self.map = [
            [0.0, 0.0],
            [5.0, 0.0]
        ]
        self.current_goal_idx = 0
        
        self.goal = [0.0, 0.0]
        self.odom_x, self.odom_y, self.odom_yaw = 0.0, 0.0, 0.0
        self.heading = 0.0
        self.erp_cmd = Twist()
        
    def goal_callback(self, goal):
        for data in point_cloud2.read_points(goal, field_names=['x', 'y', 'z'], skip_nans=True):
            self.goal = [data[0], data[1], 0.0]
    
    def heading_callback(self, heading):
        self.heading = heading.data
    
    def odom_callback(self, odom):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation

        self.odom_x = p.x
        self.odom_y = p.y
        _, _, self.odom_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.last_odom_time = time.time()
        
    def tf2(self):
        if self.goal is None:
            return
        goal_x_lidar = self.goal[0]
        goal_y_lidar = self.goal[1]
        
        goal_x = goal_x_lidar - DISTANCE_LIDAR_BASE
        goal_y = goal_y_lidar
        
        #### 회전행렬 필요없을거같긴함 ####
        # goal_x_robot = cos(self.heading) * goal_x + sin(self.heading) * goal_y
        # goal_y_robot = -sin(self.heading) * goal_x + cos(self.heading) * goal_y
        
        # return [goal_x_robot, goal_y_robot]
        
        return [goal_x, goal_y]
    
    def pub_erp_cmd_point(self):
        goal_x_robot, goal_y_robot = self.tf2()    
        heading_error = atan2(goal_y_robot, goal_x_robot)
        self.get_logger().info(f"Error heading: {np.rad2deg(heading_error):.2f} deg")

        if abs(np.rad2deg(heading_error)) > ANGULAR_THRESHOLD:
            self.erp_cmd.linear.x = 0.0
            if abs(np.rad2deg(heading_error)) < ANGULAR_THRESHOLD_STOP:
                self.erp_cmd.angular.z = 0.0
            else:
                self.erp_cmd.angular.z = Kp * heading_error
        else:
            self.erp_cmd.linear.x = FORWARD_SPEED
            self.erp_cmd.angular.z = 0.0
        
        self.erp_pub.publish(self.erp_cmd)
        
        
    def pub_erp_cmd_odom(self):
        #멈춰버리기
        if self.current_goal_idx >= len(self.map):
            self.erp_cmd.linear.x = 0.0
            self.erp_cmd.angular.z = 0.0
            self.erp_pub.publish(self.erp_cmd)
            self.get_logger().info("모든 목표에 도달했다옹.")
            return
        
        goal = self.map[self.current_goal_idx]
        
        #목표도달 확인하기
        dist = np.hypot(goal[0] - self.odom_x, goal[1] - self.odom_y)
        if dist < INDEX_THRESHOLD:
            self.get_logger().info(f"목표 {self.current_goal_idx}에 도달했다옹")
            self.current_goal_idx += 1
            return 
        
        dx = goal[0] - self.odom_x
        dy = goal[1] - self.odom_y
        goal_x_robot = cos(self.odom_yaw) * dx + sin(self.odom_yaw) * dy
        goal_y_robot = -sin(self.odom_yaw) * dx + cos(self.odom_yaw) * dy
        
        heading_error = atan2(goal_y_robot, goal_x_robot)
        self.get_logger().info(f"Error heading: {np.rad2deg(heading_error):.2f} deg")

        if abs(np.rad2deg(heading_error)) > ANGULAR_THRESHOLD_ODOM:
            self.erp_cmd.linear.x = 0.0
            self.erp_cmd.angular.z = Kp * heading_error
        else:
            self.erp_cmd.linear.x = FORWARD_SPEED
            self.erp_cmd.angular.z = 0.0
            
        self.erp_pub.publish(self.erp_cmd)
        
    def pub_what(self):
        now = time.time()
        if now - self.last_odom_time < 1.0:
            self.pub_erp_cmd_odom()
        else:
            self.pub_erp_cmd_point()
        
        
def main(args=None):
    rclpy.init(args=args)
    
    node = FollowPoint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('펑펑펑')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()