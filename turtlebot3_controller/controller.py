import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

import math
import time
import random

class Euler:
    def __init__(self, roll, pitch, yaw):
        self.roll = math.degrees(roll)
        self.pitch = math.degrees(pitch)
        self.yaw = math.degrees(yaw)

class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('turtlebot')

        #Subscribing to the LIDAR (360 floats)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)
        self.lidar_ranges = None

        self.subscription_ = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10)
        self.position = None
        self.orientation = None

        #Publisher to edit the turtle's speed
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.cmd_vel = Twist()
        self.stopped = False

        #Create turn timer
        self.start_turn_angle = -1
        self.desired_turn_angle = 0
        self.turn_timer = self.create_timer(0.1, self.turn_timer_callback)

    def lidar_callback(self, msg):
        #print(len(msg.ranges))

        self.lidar_ranges = msg.ranges
        front_range, front_side_range = self.check_distance()

        if not self.stopped and self.start_turn_angle == -1:
            # Move the robot forward
            self.cmd_vel.linear.x = 0.5
            self.cmd_vel_pub.publish(self.cmd_vel)
            self.get_logger().info('Moving robot forward. Dist = ' + str(front_range) + " , " + str(front_side_range))

    def odometry_callback(self, msg):
        self.position = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        self.orientation = self.quaternion_to_euler(quat.x,quat.y,quat.z,quat.w)

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
        front_side_range = min(self.lidar_ranges[44],self.lidar_ranges[314])
        all_range = min(self.lidar_ranges)

        if (front_range <= 0.55 or front_side_range <= 0.25 or all_range <= 0.1) and self.start_turn_angle == -1:
            # Stop the robot
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel)
            self.stopped = True
            if front_range <= 0.5:
                self.get_logger().info('Front obstacle detected. Rotating robot.')
            elif front_side_range <= 0.23:
                self.get_logger().info('Side obstacle detected. Rotating robot.')
            else:
                self.get_logger().info('Obstacle detected. Rotating robot.')
            self.start_turn(self.get_random_dir())
        else:
            self.stopped = False

        return front_range, front_side_range

    def turn_timer_callback(self):
        if self.start_turn_angle == -1:
            return

        if abs(self.desired_turn_angle - self.orientation.yaw) < 1:
            self.cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel)
            self.get_logger().info('Turned')
            self.start_turn_angle = -1
            self.turn_timer.cancel()
            self.check_distance()

    def start_turn(self, degrees):
        if degrees < 0:
            self.cmd_vel.angular.z = -0.35
        else:
            self.cmd_vel.angular.z = 0.35
        self.cmd_vel_pub.publish(self.cmd_vel)

        self.start_turn_angle = self.orientation.yaw
        #print(self.start_turn_angle, (self.start_turn_angle + degrees))

        self.desired_turn_angle = (self.start_turn_angle + degrees) % 360
        if self.desired_turn_angle < 0:
            self.desired_turn_angle += 360

        #print(self.desired_turn_angle, self.orientation.yaw)
        self.turn_timer.reset()

    def get_random_dir(self):
        dir = random.randint(90,180)
        if random.randint(0,1) == 0:
            dir *= -1
        print(dir)
        return dir


def main(args=None):
    rclpy.init(args=args)

    turtlebot = LidarSubscriber()

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        # Catch the KeyboardInterrupt exception when CTRL+C is pressed
        # Send a message to the reset_simulation service to reset the simulation
        client = turtlebot.create_client(Empty, '/reset_simulation')
        while not client.wait_for_service(timeout_sec=1.0):
            turtlebot.get_logger().info('Waiting for reset_simulation service...')
        request = Empty.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(turtlebot, future)
        turtlebot.get_logger().info('Simulation reset.')

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
