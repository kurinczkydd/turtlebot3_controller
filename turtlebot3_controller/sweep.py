import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray, Int32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import copy
import numpy as np
from collections import deque
from operator import itemgetter
from enum import Enum
import math
import datetime
import time

class Dir(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

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

class SweepNode(Node):

    def __init__(self):
        super().__init__('sweep')

        #Subscribing to the Odometry infos
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10)
        self.position = None

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
        # 7 ToSweep

        self.path_sub = self.create_subscription(
            Int32MultiArray,
            '/path_sweep',
            self.path_callback,
            10)
        
        self.path_pub = self.create_publisher(
            Int32MultiArray,
            '/path',
            10)
        
        self.marker_sweep_pub = self.create_publisher(
            Marker,
            '/path_sweep_marker',
            10)
        
        self.map_data_sub = self.create_subscription(
            Int32MultiArray,
            '/map_data',
            self.map_data_callback,
            10)
        
        self.goal_pub = self.create_publisher(
            Int32MultiArray,
            '/goal_sweep',
            10)

        #SLAM map
        self.mapPosition = None
        self.mapSize = (320,320)
        self.mapRes = 50
        self.mapLimits = DirLimit(320,0,320,0)
        self.map = None

        #Path starts and goals for PathNode
        self.pathNodeCoords = []
        self.pathNodeCurrent = 0
        self.fullPath = []
        self.pathConnecting = []

    def map_data_callback(self, msg):
        self.mapLimits = DirLimit(msg.data[0],msg.data[1],msg.data[2],msg.data[3])
        self.map = [msg.data[i:i+320] for i in range(4, len(msg.data), 320)]

    def odometry_callback(self, msg):
        self.position = msg.pose.pose.position
        px = self.mapSize[0]//2 + int(round(self.position.x,5) * self.mapRes)
        py = self.mapSize[1]//2 + int(round(self.position.y,5) * self.mapRes)
        self.mapPosition = (px,py)

    def path_callback(self, msg):

        self.pathConnecting.append([msg.data[i:i+2] for i in range(0, len(msg.data), 2)])
        self.get_logger().info("Got AStar path (" + str(self.pathNodeCurrent+1) + "/" + str(len(self.pathNodeCoords)) + ")")
        self.pathNodeCurrent += 1
        
        if self.pathNodeCurrent < len(self.pathNodeCoords):
            self.sendNextPathPlanToPathNode()
        else:
            path = []
            for i in range(len(self.fullPath)):
                if i < len(self.fullPath):
                    path.append(self.fullPath[i])
                if i < len(self.pathConnecting):
                    path.append(self.pathConnecting[i])

            sweepingPath = [item for sublist in path for item in sublist]
            sweepingPathFlat = [item for sublist in sweepingPath for item in sublist]

            self.turtle_state = 5 #Pathfinding state
            stateMsg = Int32()
            stateMsg.data = 5
            self.state_pub.publish(stateMsg)

            self.place_markers_sweep(sweepingPath)

            time.sleep(2)

            msg = Int32MultiArray()
            msg.data = sweepingPathFlat
            self.path_pub.publish(msg)
            self.get_logger().info("AStar path published (" + str(len(sweepingPath)) + ")")


    def turtle_state_callback(self, msg):
        self.turtle_state = msg.data

        if self.turtle_state == 6 and self.map is not None:
            self.get_logger().info("Sweeping started")
            mapLimits = self.mapLimits.getAsArray()
            map_data = self.processMapForSweep(self.map, 10, mapLimits)

            self.get_logger().info(str(mapLimits[0]) + " " + str(mapLimits[1]) + " " + str(mapLimits[2]) + " " + str(mapLimits[3]))
            bounds = self.getRoomBoundaries(map_data, mapLimits)
            self.get_logger().info(str(bounds[0]) + " " + str(bounds[1]) + " " + str(bounds[2]) + " " + str(bounds[3]))
            processedMap = self.processMap(map_data, bounds)

            processedMap, path = self.getPath(processedMap, (self.mapPosition[0],self.mapPosition[1]), (self.mapSize[0]-1, self.mapSize[1]-1), bounds)
            self.fullPath = [path]
            self.get_logger().info("Got region 1")

            regions = self.find_regions(processedMap, (bounds[0]+5, bounds[1]-5, bounds[2]+5, bounds[3]-5), 0)
            while self.isThereMoreRegions(regions, 1000):
                self.get_logger().info("Sweeping region " + str(len(self.fullPath)+1))
                processedMap, path = self.getPath(processedMap, (regions[0][0][0],regions[0][0][1]), (self.mapSize[0]-1, self.mapSize[1]-1), bounds)
                self.fullPath.append(path)
                self.get_logger().info("Got region " + str(len(self.fullPath)))
                regions = self.find_regions(processedMap, (bounds[0]+5, bounds[1]-5, bounds[2]+5, bounds[3]-5), 0)
                self.get_logger().info("Got next regions " + str(len(self.fullPath)+1))

            self.get_logger().info("Finished sweeping regions, connecting them")
                
            #got all sweep paths
            self.getAllCoordsForPathNode()
            self.sendNextPathPlanToPathNode()
            

    def sendNextPathPlanToPathNode(self):
        self.turtle_state = 7 #AStar sweeping state
        stateMsg = Int32()
        stateMsg.data = 7
        self.state_pub.publish(stateMsg)

        time.sleep(2)

        msg = Int32MultiArray()
        coords = self.pathNodeCoords[self.pathNodeCurrent]
        msg.data = [coords[0][0],coords[0][1],coords[1][0], coords[1][1]]
        self.goal_pub.publish(msg)
        self.get_logger().info("Sweeping path goal published (" + str(self.pathNodeCurrent+1) + "/" + str(len(self.pathNodeCoords)) + ")")

    def getAllCoordsForPathNode(self):
        self.pathNodeCoords = []
        self.pathNodeCurrent = 0

        start = None
        end = self.mapPosition
        for i in range(1, len(self.fullPath)):
            start = self.fullPath[i][0]
            self.pathNodeCoords.append([start, end])
            end = self.fullPath[i][-1]

    def processMapForSweep(self, grid, pad_distance, mapLimits):
        new_grid = copy.deepcopy(grid)
        # iterate over the cells in the grid
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                # if this cell is a wall
                if grid[i][j] == 2:
                    if i < mapLimits[0] or i > mapLimits[1] or j < mapLimits[2] or j > mapLimits[3]:
                        new_grid[i][j] = 0
                    else:
                        # iterate over the neighboring cells within the padding distance
                        for di in range(-pad_distance, pad_distance + 1):
                            for dj in range(-pad_distance, pad_distance + 1):
                                # if the neighboring cell is within the grid and not a wall
                                ni, nj = i + di, j + dj
                                if (mapLimits[0] <= ni < mapLimits[1]) and (mapLimits[2] <= nj < mapLimits[3]) and grid[ni][nj] != 2:
                                    # set it to be a wall
                                    new_grid[ni][nj] = 2
        return new_grid

    def isPathFree(self, mat, pos, dir):
        if dir == Dir.UP:
            if pos[0]-1 < 0:
                return False
            else:
                return mat[pos[0]-1][pos[1]] != 1 and mat[pos[0]-1][pos[1]] != 2
        elif dir == Dir.DOWN:
            if pos[0]+1 >= len(mat):
                return False
            else:
                return mat[pos[0]+1][pos[1]] != 1 and mat[pos[0]+1][pos[1]] != 2
        elif dir == Dir.LEFT:
            if pos[1]-1 < 0:
                return False
            else:
                if mat[pos[0]][pos[1]-1] != 1 and mat[pos[0]][pos[1]-1] != 2:
                    return True
                else:
                    return self.isPathFree(mat, (pos[0], pos[1]-1), Dir.UP)
        else: #RIGHT
            if pos[1]+1 >= len(mat[0]):
                return False
            else:
                if mat[pos[0]][pos[1]+1] != 1 and mat[pos[0]][pos[1]+1] != 2:
                    return True
                else:
                    return self.isPathFree(mat, (pos[0], pos[1]+1), Dir.UP)
                
    def isPathFreeBounding(self, bounding, pos, dir):
        if bounding is None:
            return True

        if dir == Dir.UP:
            return pos[0]-1 > bounding[0]
        elif dir == Dir.DOWN:
            return pos[0]+1 < bounding[1]
        elif dir == Dir.LEFT:
            return pos[1]-1 > bounding[2]
        else: #RIGHT
            return pos[1]+1 < bounding[3]
        
    def goPath(self, pos, dir):
        if dir == Dir.UP:
            return (pos[0]-1, pos[1])
        elif dir == Dir.DOWN:
            return (pos[0]+1, pos[1])
        elif dir == Dir.LEFT:
            return (pos[0], pos[1]-1)
        else: #RIGHT
            return (pos[0], pos[1]+1)
        
    def rotateDir(self, dir, prevDir):
        if prevDir == None:
            return (Dir.RIGHT, Dir.DOWN)

        if dir == Dir.UP:
            pass
        elif dir == Dir.DOWN and prevDir == Dir.LEFT:
            return (Dir.DOWN, Dir.RIGHT)
        elif dir == Dir.DOWN and prevDir == Dir.RIGHT:
            return (Dir.DOWN, Dir.LEFT)
        elif dir == Dir.LEFT:
            return (Dir.LEFT, Dir.DOWN)
        else: #Right
            return (Dir.RIGHT, Dir.DOWN)
        
    def resetSweep(self, mat):
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == 3:
                    mat[i][j] = 2
        return mat

    def getPath(self, mat, start, goal, bounding, radius = 7):
        a = datetime.datetime.now()

        prevDir = None
        prevLength = 0

        dir = Dir.RIGHT

        path = []
        
        current = start

        while current != goal:
            if self.isPathFree(mat, current, dir) and self.isPathFreeBounding(bounding, current, dir):
                if dir == Dir.DOWN:
                    prevLength += 1

                current = self.goPath(current, dir)

                if path.count(current) > 2:
                    mat = self.resetSweep(mat)
                    return mat, path[:len(path)-10]

                path.append(current)
                for i in range(-radius, radius+1):
                    for j in range(-radius, radius+1):
                        if mat[current[0]+i][current[1]+j] != 1 and mat[current[0]+i][current[1]+j] != 2:
                            mat[current[0]+i][current[1]+j] = 3

                if prevLength >= radius*2:
                    prevLength = 0
                    prevDir, dir = self.rotateDir(dir, prevDir)
            else: #Hit wall
                prevDir, dir = self.rotateDir(dir, prevDir)
                prevLength = 0

            b = (datetime.datetime.now()-a)
            if b.seconds > 30:
                o = "paths: "
                for p in path:
                    o += str(p[0]) + " " + str(p[1]) + " -> "
                self.get_logger().info(o)
                mat = self.resetSweep(mat)
                return mat, path

        mat = self.resetSweep(mat)
        return mat, path
    
    def remove_outliers(self, data, n=2):
        mean = np.mean(data)
        std_dev = np.std(data)
        
        return [x for x in data if (x > mean - n * std_dev) and (x < mean + n * std_dev)]

    def getRoomBoundaries(self, mat, mapLimit, freq=10, backUpRange=15, excludeRange = -25):

        boundaries = []
        #UP
        avg = []
        for i in range(freq-1):
            step = (mapLimit[3]-mapLimit[2])/freq
            x = int(mapLimit[2] + (step*(i+1)))
            y = mapLimit[0]-backUpRange

            while mat[y][x] != 2:
                y += 1
            while mat[y][x] == 2:
                y += 1
            y += excludeRange
            avg.append(y)
        avg = self.remove_outliers(avg)
        a = sum(avg)/len(avg)
        boundaries.append(a)

        #DOWN
        avg = []
        for i in range(freq-1):
            step = (mapLimit[3]-mapLimit[2])/freq
            x = int(mapLimit[2] + (step*(i+1)))
            y = mapLimit[1]+backUpRange

            while mat[y][x] != 2:
                y -= 1
            while mat[y][x] == 2:
                y -= 1
            y -= excludeRange
            avg.append(y)
        avg = self.remove_outliers(avg)
        a = sum(avg)/len(avg)
        boundaries.append(a)

        #LEFT
        avg = []
        for i in range(freq-1):
            step = (mapLimit[1]-mapLimit[0])/freq
            x = mapLimit[2]-backUpRange
            y = int(mapLimit[0] + (step*(i+1)))

            while mat[y][x] != 2:
                x += 1
            while mat[y][x] == 2:
                x += 1
            x += excludeRange
            avg.append(x)
        avg = self.remove_outliers(avg)
        a = sum(avg)/len(avg)
        boundaries.append(a)

        #RIGHT
        avg = []
        for i in range(freq-1):
            step = (mapLimit[1]-mapLimit[0])/freq
            x = mapLimit[3]+backUpRange
            y = int(mapLimit[0] + (step*(i+1)))

            while mat[y][x] != 2:
                x -= 1
            while mat[y][x] == 2:
                x -= 1
            x -= excludeRange
            avg.append(x)
        avg = self.remove_outliers(avg)
        a = sum(avg)/len(avg)
        boundaries.append(a)

        return boundaries
    
    def processMap(self, mat, bounds):
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if i < bounds[0] or i > bounds[1] or j < bounds[2] or j > bounds[3]:
                    mat[i][j] = 0
                elif mat[i][j] == 1:
                    mat[i][j] = 0

        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == 2:
                    mat[i][j] = 1
        return mat

    def isThereMoreRegions(self, regions, size=25):
        for region in regions:
            if region[1] >= size:
                return True
            else:
                return False

    def find_regions(self, map, bounding, cell=1):
        map = np.array(map)
        height, width = map.shape
        visited = np.zeros((height, width), dtype=bool)
        dx = [0, 0, -1, 1]
        dy = [-1, 1, 0, 0]
        regions = []

        def is_valid(x, y):
            return bounding[2] <= x < bounding[3] and bounding[0] <= y < bounding[1]

        def bfs(start):
            queue = deque([start])
            visited[start[0], start[1]] = True
            region_size = 0
            while queue:
                x, y = queue.popleft()
                region_size += 1
                for direction in range(4):
                    nx, ny = x + dx[direction], y + dy[direction]
                    if is_valid(nx, ny) and not visited[nx][ny] and map[nx][ny] == cell:
                        queue.append((nx, ny))
                        visited[nx, ny] = True
            return region_size

        for i in range(int(bounding[0]),int(bounding[1])):
            for j in range(int(bounding[2]),int(bounding[3])):
                if map[i][j] == cell and not visited[i][j]:
                    regions.append(((i, j), bfs((i, j))))
                    
        regions.sort(key=itemgetter(1), reverse=True)

        return [region for region in regions]
    


    def place_markers_sweep(self, positions):

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

        self.marker_sweep_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)

    turtlebot = SweepNode()

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        pass

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()