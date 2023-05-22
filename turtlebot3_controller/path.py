import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import math
import time
import numpy as np
import heapq
import copy

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
    

def pad_walls(grid, pad_distance):
    # create a copy of the grid to avoid modifying it while iterating
    new_grid = copy.deepcopy(grid)
    
    # iterate over the cells in the grid
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # if this cell is a wall
            if grid[i][j] == 2:
                # iterate over the neighboring cells within the padding distance
                for di in range(-pad_distance, pad_distance + 1):
                    for dj in range(-pad_distance, pad_distance + 1):
                        # if the neighboring cell is within the grid and not a wall
                        ni, nj = i + di, j + dj
                        if (0 <= ni < len(grid)) and (0 <= nj < len(grid[i])) and grid[ni][nj] != 2:
                            # set it to be a wall
                            new_grid[ni][nj] = 2
    return new_grid

def clearRadius(map, i, j, radius=15):
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            map[i+di][j+dj] = 1

# Our priority queue
class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def a_star_search(map, start, goal):
    if map[start[0]][start[1]] == 2:
        clearRadius(map, start[0], start[1], 5)

    if map[goal[0]][goal[1]] == 2:
        clearRadius(map, goal[0], goal[1], 5)

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in neighbors(map, current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    if tuple(goal) not in came_from:
        return None

    # Reconstruct the path
    current = tuple(goal)
    path = []
    while current != tuple(start):
        path.append(current)
        current = came_from[current]
    path.append(tuple(start))  # optional
    path.reverse()  # optional
    return path


# Get the neighbors of a node
def neighbors(map, node):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    result = []
    for dir in directions:
        neighbor = (node[0] + dir[0], node[1] + dir[1])
        if neighbor[0] >= 0 and neighbor[0] < map.shape[0] and neighbor[1] >= 0 and neighbor[1] < map.shape[1]: # Within map
            if map[neighbor[0]][neighbor[1]] != 2:  # Not a wall
                result.append(tuple(neighbor))  # ensure the neighbor is a tuple
    return result

# Heuristic function
def heuristic(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

class PathNode(Node):

    def __init__(self):
        super().__init__('path')

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
        self.mapSize = (320,320)
        self.mapRes = 50
        self.mapLimits = DirLimit(320,0,320,0)
        self.map = np.zeros(self.mapSize,dtype=int)

        #Publisher to visualization marker
        self.marker_pub = self.create_publisher(
            Marker,
            '/path_marker',
            10)
        
        self.goal_marker_pub = self.create_publisher(
            Marker,
            '/goal_marker',
            10)
        self.deleteRvizMarkers()

    def turtle_state_callback(self, msg):
        self.turtle_state = msg.data

    def odometry_callback(self, msg):
        self.position = msg.pose.pose.position
        px = self.mapSize[0]//2 + int(round(self.position.x,5) * self.mapRes)
        py = self.mapSize[1]//2 + int(round(self.position.y,5) * self.mapRes)
        self.mapPosition = (px,py)

    def goal_callback(self, msg):
        if self.turtle_state == 4:
            goals = [msg.data[i:i+2] for i in range(0, len(msg.data), 2)]

            padded_map = pad_walls(self.map,10)
            padded_map = np.array(padded_map)

            for goal in goals:
                self.place_marker(goal)
                self.get_logger().info("Finding path from " + str(self.mapPosition[0]) + " " + str(self.mapPosition[1]) \
                                       + " to " + str(goal[0]) + " " + str(goal[1]))

                #a_star_search
                self.search_path = a_star_search(padded_map, self.mapPosition, goal)

                if self.search_path is not None:
                    if len(self.search_path) > 1:
                        break

                
            if self.search_path is None or len(self.search_path) == 0:
                self.get_logger().info("Failed to find any AStar routes!")
                with open("/home/chloe/turtlebot3_ws/src/turtlebot3_controller/map.txt","w") as file:
                    for x in range(len(self.map)):
                        for y in range(len(self.map[0])):
                            file.write(str(self.map[x][y]) + " ")
                        file.write("\n")
                self.get_logger().info("Done saving map")

                self.deleteRvizMarkers()

                self.turtle_state = 1 #Wait for command state
                stateMsg = Int32()
                stateMsg.data = 1
                self.state_pub.publish(stateMsg)
            else:
                print("PATH: ", self.search_path[0], self.search_path[-1])
                #self.search_path.reverse()
                path = [item for sublist in self.search_path for item in sublist]

                self.turtle_state = 5 #Pathfinding state
                stateMsg = Int32()
                stateMsg.data = 5
                self.state_pub.publish(stateMsg)

                self.place_markers(self.search_path)

                time.sleep(1)

                msg = Int32MultiArray()
                msg.data = path
                self.path_pub.publish(msg)
                self.get_logger().info("AStar path published (" + str(len(self.search_path)) + ")")

    def map_data_callback(self, msg):
        self.mapLimits = DirLimit(msg.data[0],msg.data[1],msg.data[2],msg.data[3])
        self.map = [msg.data[i:i+320] for i in range(4, len(msg.data), 320)]

    def deleteRvizMarkers(self):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.action = Marker.DELETEALL
        self.marker_pub.publish(marker)

        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 1
        marker.action = Marker.DELETEALL
        self.goal_marker_pub.publish(marker)

    def place_marker(self, goal):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        px = (goal[0] - self.mapSize[0]//2) / self.mapRes
        py = (goal[1] - self.mapSize[1]//2) / self.mapRes
        goal_pos = Point()
        goal_pos.x = float(px)
        goal_pos.y = float(py)
        goal_pos.z = 0.0

        marker.pose.position = goal_pos

        self.goal_marker_pub.publish(marker)

    def place_markers(self, positions):

        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line strip is blue
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Line strip width
        marker.scale.x = 0.1

        # Add points to the line strip
        for coords in positions:
            px = (coords[0] - self.mapSize[0]//2) / self.mapRes
            py = (coords[1] - self.mapSize[1]//2) / self.mapRes

            point = Point()
            point.x = float(px)
            point.y = float(py)
            point.z = 0.0
            marker.points.append(point)

        self.marker_pub.publish(marker)

        """
        markers = []
        for pos in positions:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = self.marker_id
            self.marker_id += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.1
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position = pos
            markers.append(marker)

        marker_array = MarkerArray()
        marker_array.markers = markers
        self.marker_array_pub.publish(marker_array)
        """


def main(args=None):
    rclpy.init(args=args)

    turtlebot = PathNode()

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        pass

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
