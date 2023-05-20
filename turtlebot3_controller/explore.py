import stat
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32

import math
import numpy as np
import time

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

class DirLimit:
    def __init__(self, up,down,left,right):
        self.up = up
        self.down = down
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.up) + " " + str(self.down) + " " + str(self.left) + " " + str(self.right)
    
    def getAsArray(self):
        return [self.up, self.down, self.left, self.right]

class ExploreNode(Node):

    def __init__(self):
        super().__init__('explorenode')

        #Subscribing to the map_data
        self.map_data_sub = self.create_subscription(
            Int32MultiArray,
            '/map_data',
            self.map_data_callback,
            10)
        
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

        self.goal_pub = self.create_publisher(
            Int32MultiArray,
            '/goal',
            10)

        #SLAM map
        self.mapLimits = DirLimit(320,0,320,0)
        self.map = np.zeros(self.mapSize,dtype=int)

    def odometry_callback(self, msg):
        self.position = msg.pose.pose.position

    def turtle_state_callback(self, msg):
        self.turtle_state = msg.data

        if self.turtle_state == 2:
            self.turtle_state = 4 #AStar state
            stateMsg = Int32()
            stateMsg.data = 4
            self.state_pub.publish(stateMsg)

            time.sleep(1)

            undiscovered = self.find_largest_block()
            msg = Int32MultiArray()
            msg.data = undiscovered
            self.goal_pub.publish(msg)
            self.get_logger().info("Undiscovered area goal published")

    def map_data_callback(self, msg):
        self.mapLimits = DirLimit(msg.data[0],msg.data[1],msg.data[2],msg.data[3])
        self.map = [msg.data[i:i+320] for i in range(4, len(msg.data), 320)]

    def find_largest_block(self):
        rows = len(self.map)
        cols = len(self.map[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        max_count = 0
        start_pos = None

        border = 10

        for i in range(self.mapLimits.up + border, self.mapLimits.down + 1 - border):
            for j in range(self.mapLimits.left + border, self.mapLimits.right + 1 - border):
                if self.map[i][j] == 0 and not visited[i][j]:
                    count = 0
                    stack = [(i, j)]  # Using a stack to simulate recursion

                    while stack:
                        x, y = stack.pop()
                        if not visited[x][y] and self.map[x][y] == 0:
                            visited[x][y] = True
                            count += 1

                            # Add neighboring undiscovered zeros to the stack
                            if x + 1 < self.mapLimits.down - border and self.map[x+1][y] == 0:
                                stack.append((x + 1, y))  # Down
                            if x - 1 >= self.mapLimits.up + border and self.map[x-1][y] == 0:
                                stack.append((x - 1, y))  # Up
                            if y + 1 < self.mapLimits.left - border and self.map[x][y+1] == 0:
                                stack.append((x, y + 1))  # Right
                            if y - 1 >= self.mapLimits.right + border and self.map[x][y-1] == 0:
                                stack.append((x, y - 1))  # Left

                    if count > max_count:
                        max_count = count
                        start_pos = (i, j)

        return [start_pos[0],start_pos[1]]

def main(args=None):
    rclpy.init(args=args)

    turtlebot = ExploreNode()

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        turtlebot.stop()

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
