import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray, Int32

import math
import time
import random
import numpy as np

class Euler:
    def __init__(self, roll, pitch, yaw):
        self.roll = math.degrees(roll)
        self.pitch = math.degrees(pitch)
        self.yaw = math.degrees(yaw)

    def __str__(self):
        return "roll: " + str(round(self.roll,2)) + ", pitch: " + str(round(self.pitch,2)) + ", yaw: " + str(round(self.yaw,2))

class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        if isinstance(other, Pos):
            return self.x == other.x and self.y == other.y
        return False
    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y)

class ControllerNode(Node):

    def __init__(self):
        super().__init__('controllernode')

        #Subscribing to the LIDAR (360 floats)
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)
        self.lidar_ranges = None

        #Subscribing to the Odometry infos
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10)
        self.position = None
        self.orientation = None

        #Publisher to edit the turtle's speed and rotation
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10)
        self.cmd_vel = Twist()
        self.stopped = False

        self.state_sub = self.create_subscription(
            Int32,
            '/turtle_state',
            self.turtle_state_callback,
            10)
        
        self.state_pub = self.create_publisher(
            Int32,
            '/turtle_state',
            10)
        self.turtle_state = 0
        # 0 Auto Route
        # 1 Wait for command
        # 2 Get route
        # 3 Stop
        # 4 AStar
        # 5 Pathfinding

        self.path_pub = self.create_subscription(
            Int32MultiArray,
            '/path',
            self.path_callback,
            10)

        #Create turn timer
        self.start_turn_angle = -1
        self.desired_turn_angle = 0
        self.turn_timer = self.create_timer(0.1, self.turn_timer_callback)

        #SLAM map
        self.mapPosition = None

        self.firstWall = True
        self.search_path = None
        self.search_next_step = 5

    def stop(self):
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(self.cmd_vel)

    def turtle_state_callback(self, msg):
        self.turtle_state = msg.data

    def path_callback(self, msg):
        self.search_path = msg.data

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges
        front_range, front_side_range = self.check_distance()

        # 0 Auto Route
        # 1 Wait for command
        # 2 Get route
        # 3 Stop
        # 4 AStar
        # 5 Pathfinding

        if self.turtle_state == 0: #Auto route
            if not self.stopped and self.start_turn_angle == -1:
                self.cmd_vel.linear.x = 0.5
                self.cmd_vel_pub.publish(self.cmd_vel)
                self.get_logger().info('Moving robot forward. Dist = ' + str(front_range) + " , " + str(front_side_range))
        elif self.turtle_state == 1: # Wait for command
            pass
        elif self.turtle_state in [2,3,4]: # Get route, Stop, AStar
            if not self.stopped:
                self.stop()
        elif self.turtle_state == 5 and self.search_path is not None and self.mapPosition is not None: # Pathfinding
            dirToGoal = self.calculate_angle(self.mapPosition, self.search_path[self.search_next_step])

            if abs(self.orientation.yaw - dirToGoal) > 1.0:
                self.stop()
                self.start_turn(dirToGoal)
            elif self.heuristic(self.mapPosition, self.search_path[self.search_next_step]) <= 1:
                self.search_next_step += 1
                if self.search_next_step >= len(self.search_path):
                    self.get_logger().info("Got to the AStar path, finding another.")
                    self.search_next_step = 5
                    self.search_path = None

                    msg = Int32()
                    msg.data = 2
                    self.state_pub.publish(msg)
                    self.turtle_state = 2
            else:
                self.cmd_vel.linear.x = 0.5
                self.cmd_vel_pub.publish(self.cmd_vel)
                self.get_logger().info('Moving robot forward. Dist = ' + str(front_range) + " , " + str(front_side_range))


    def odometry_callback(self, msg):
        self.position = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        self.orientation = self.quaternion_to_euler(quat.x,quat.y,quat.z,quat.w)

        px = len(self.map)//2 + int(round(self.position.x,5) * self.mapRes)
        py = len(self.map[0])//2 + int(round(self.position.y,5) * self.mapRes)
        self.mapPosition = (px,py)

        if abs(self.orientation.roll) >= 5:
            self.get_logger().info('FALLING OVER!! roll:' + str(self.orientation.roll))
        if abs(self.orientation.pitch) >= 5:
            self.get_logger().info('FALLING OVER!! pitch:' + str(self.orientation.pitch))

    def calculate_angle(self,start, goal):
        x1,y1 = start[0],start[1]
        x2,y2 = goal[0],goal[1]

        dx = x2 - x1
        dy = y2 - y1
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees
    
    def delta_degrees(self, angle1, angle2):
        delta = (angle2 - angle1) % 360
        if delta > 180:
            delta -= 360
        return delta
    
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def quaternion_to_euler(self, x, y, z, w):
        # Convert the quaternion to Euler angles (roll, pitch, yaw)
        # See https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        if yaw < 0:
            yaw += 2 * math.pi

        return Euler(roll, pitch, yaw)

    def check_distance(self):
        front_range = self.lidar_ranges[0]
        front_side_range = min(self.lidar_ranges[29],self.lidar_ranges[329])
        all_range = min(self.lidar_ranges)

        if (front_range <= 0.4 or front_side_range <= 0.3 or all_range <= 0.1) and self.start_turn_angle == -1:

            if self.firstWall:
                msg = Int32()
                msg.data = 2
                self.state_pub.publish(msg)
                self.turtle_state = 2
                self.firstWall = False
            else:
                # Stop the robot
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel_pub.publish(self.cmd_vel)
                self.stopped = True
                if front_range <= 0.5:
                    self.get_logger().info('Front obstacle detected. Rotating robot.')
                elif front_side_range <= 0.3:
                    self.get_logger().info('Side obstacle detected. Rotating robot.')
                else:
                    self.get_logger().info('Obstacle detected. Rotating robot.')
                
                farthestAngle = self.lidar_ranges.index(max(self.lidar_ranges))
                self.start_turn(self.delta_degrees(self.orientation.yaw, farthestAngle))
        else:
            self.stopped = False

        return front_range, front_side_range

    def turn_timer_callback(self):
        if self.start_turn_angle == -1:
            return

        if abs(self.desired_turn_angle - self.orientation.yaw) < 1:
            self.cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel)
            self.start_turn_angle = -1
            self.turn_timer.cancel()
            self.check_distance()

    def start_turn(self, degrees):
        if self.orientation is None:
            return

        if degrees < 0:
            self.cmd_vel.angular.z = -0.35
        else:
            self.cmd_vel.angular.z = 0.35
        self.cmd_vel_pub.publish(self.cmd_vel)

        self.start_turn_angle = self.orientation.yaw

        self.desired_turn_angle = (self.start_turn_angle + degrees) % 360
        if self.desired_turn_angle < 0:
            self.desired_turn_angle += 360

        self.get_logger().info('Turning ' + str(degrees) + " degrees")
        self.turn_timer.reset()

    def get_random_dir(self):
        dir = random.randint(90,140)
        if random.randint(0,1) == 0:
            dir *= -1
        print(dir)
        return dir

def main(args=None):
    rclpy.init(args=args)

    turtlebot = ControllerNode()

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        turtlebot.stop()

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
