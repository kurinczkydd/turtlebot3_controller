import stat
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32

from collections import deque
from operator import itemgetter
import math
import numpy as np
import time
import copy

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

class ExploreNode(Node):

    def __init__(self):
        super().__init__('explore')

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
        # 6 Sweeping

        self.goal_pub = self.create_publisher(
            Int32MultiArray,
            '/goal',
            10)

        #SLAM map
        self.mapSize = (320,320)
        self.mapRes = 50
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

            time.sleep(2)

            #undiscovered = self.find_regions(np.array(self.map))
            undiscovered = self.find_all_groups(pad_walls(self.map, 6))

            undiscGood = []
            for und in undiscovered:
                if und[1] >= 500:
                    undiscGood.append(und[0])

            if len(undiscGood) > 0:
                msg = Int32MultiArray()
                msg.data = [item for sublist in undiscGood for item in sublist]
                self.goal_pub.publish(msg)
                self.get_logger().info("Undiscovered area goals published (" + str(len(undiscGood)) + ")")
            else:
                self.get_logger().info("Cannot find any undiscovered area, sweeping")
                self.turtle_state = 6 #Sweep state
                stateMsg = Int32()
                stateMsg.data = 6
                self.state_pub.publish(stateMsg)

    def map_data_callback(self, msg):
        self.mapLimits = DirLimit(msg.data[0],msg.data[1],msg.data[2],msg.data[3])
        self.map = [msg.data[i:i+320] for i in range(4, len(msg.data), 320)]

    def find_regions(self, map):
        height, width = map.shape
        visited = np.zeros((height, width), dtype=bool)
        dx = [0, 0, -1, 1]
        dy = [-1, 1, 0, 0]
        regions = []

        def is_valid(x, y):
            return 0 <= x < height and 0 <= y < width

        def bfs(start):
            queue = deque([start])
            visited[start[0], start[1]] = True
            region_size = 0
            while queue:
                x, y = queue.popleft()
                region_size += 1
                for direction in range(4):
                    nx, ny = x + dx[direction], y + dy[direction]
                    if is_valid(nx, ny) and not visited[nx][ny] and map[nx][ny] == 0:
                        queue.append((nx, ny))
                        visited[nx, ny] = True
            return region_size

        for i in range(height):
            for j in range(width):
                if map[i][j] == 0 and not visited[i][j]:
                    regions.append(((i, j), bfs((i, j))))
                    
        regions.sort(key=itemgetter(1), reverse=True)

        return [region for region in regions]

    def find_all_groups(self, map):
        rows = len(map)
        cols = len(map[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        groups = []
        border = 10
    
        for i in range(self.mapLimits.up + border, self.mapLimits.down + 1 - border):
            for j in range(self.mapLimits.left + border, self.mapLimits.right + 1 - border):
                if map[i][j] == 0 and not visited[i][j]:
                    count = 0
                    stack = [(i, j)]  # Using a stack to simulate recursion
    
                    while stack:
                        x, y = stack.pop()
                        if not visited[x][y] and map[x][y] == 0:
                            visited[x][y] = True
                            count += 1
    
                            # Add neighboring undiscovered zeros to the stack
                            if x + 1 < self.mapLimits.down - border and map[x+1][y] == 0:
                                stack.append((x + 1, y))  # Down
                            if x - 1 >= self.mapLimits.up + border and map[x-1][y] == 0:
                                stack.append((x - 1, y))  # Up
                            if y + 1 < self.mapLimits.right - border and map[x][y+1] == 0:
                                stack.append((x, y + 1))  # Right
                            if y - 1 >= self.mapLimits.left + border and map[x][y-1] == 0:
                                stack.append((x, y - 1))  # Left
    
                    # We've finished exploring this group, so add the starting position and size to the list
                    groups.append(([i, j], count))
    
        # Sort the groups by size in descending order
        groups.sort(key=lambda x: x[1], reverse=True)
    
        # Return a list of starting positions, in descending order of group size
        return [group for group in groups]


def main(args=None):
    rclpy.init(args=args)

    turtlebot = ExploreNode()

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        pass

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
