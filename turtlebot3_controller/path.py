import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32
from nav_msgs.msg import Odometry

import math
import time
import numpy as np
import heapq

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

class DirLimit:
    def __init__(self, up,down,left,right):
        self.up = up
        self.down = down
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.up) + " " + str(self.down) + " " + str(self.left) + " " + str(self.right)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def in_bounds(grid, node):
    return 0 <= node[0] < len(grid) and 0 <= node[1] < len(grid[0])

def is_traversable(grid, node):
    return grid[node[0]][node[1]] in (0, 1)

def get_clearance(grid, node, clearance):
    for i in range(-clearance, clearance + 1):
        for j in range(-clearance, clearance + 1):
            check_node = (node[0] + i, node[1] + j)
            if in_bounds(grid, check_node) and grid[check_node[0]][check_node[1]] == 2:
                return False
    return True

def get_neighbors(grid, current, goal, clearance=15):
    neighbors = [(current[0] + d[0], current[1] + d[1]) for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
    valid_neighbors = []
    heuristic_value = heuristic(current, goal)
    current_clearance = clearance if heuristic_value > 5 else 0

    for neighbor in neighbors:
        if in_bounds(grid, neighbor) and is_traversable(grid, neighbor) and get_clearance(grid, neighbor, current_clearance):
            valid_neighbors.append(neighbor)
    return valid_neighbors

def a_star(grid, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for next in get_neighbors(grid, current, goal):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    if current != goal:
        return None

    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

class PathNode(Node):

    def __init__(self):
        super().__init__('pathnode')

        #Subscribing to the Odometry infos
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10)
        self.position = None

        self.goal_sub = self.create_subscription(
            Int32MultiArray,
            '/goal',
            self.goal_callback,
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
        
        self.map_data_sub = self.create_subscription(
            Int32MultiArray,
            '/map_data',
            self.map_data_callback,
            10)

        self.path_pub = self.create_publisher(
            Int32MultiArray,
            '/path',
            10)

        #SLAM map
        self.mapPosition = None
        self.mapLimits = DirLimit(320,0,320,0)
        self.map = np.zeros(self.mapSize,dtype=int)

    def turtle_state_callback(self, msg):
        self.turtle_state = msg.data

    def odometry_callback(self, msg):
        self.position = msg.pose.pose.position
        px = len(self.map)//2 + int(round(self.position.x,5) * self.mapRes)
        py = len(self.map[0])//2 + int(round(self.position.y,5) * self.mapRes)
        self.mapPosition = (px,py)

    def goal_callback(self, msg):
        if self.turtle_state == 4:
            goal = (msg.data[0],msg.data[1])
            self.search_path = a_star(self.map, self.mapPosition, goal)
            path = [item for sublist in self.search_path for item in sublist]

            self.turtle_state = 5 #Pathfinding state
            stateMsg = Int32()
            stateMsg.data = 5
            self.state_pub.publish(stateMsg)

            time.sleep(1)

            msg = Int32MultiArray()
            msg.data = path
            self.path_pub.publish(msg)
            self.get_logger().info("AStar path published")

    def map_data_callback(self, msg):
        self.mapLimits = DirLimit(msg.data[0],msg.data[1],msg.data[2],msg.data[3])
        self.map = [msg.data[i:i+320] for i in range(4, len(msg.data), 320)]


def main(args=None):
    rclpy.init(args=args)

    turtlebot = PathNode()

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        turtlebot.stop()

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
