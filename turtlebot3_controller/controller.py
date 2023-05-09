import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import GridCells
from nav_msgs.msg import OccupancyGrid

import math
import time
import random
import numpy as np
import os
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

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def in_bounds(grid, node):
    return 0 <= node[0] < len(grid) and 0 <= node[1] < len(grid[0])

def is_traversable(grid, node):
    return grid[node[0]][node[1]] in (0, 1)

#def get_neighbors(grid, current, wall_buffer=5):
#    neighbors = [(current[0] + d[0], current[1] + d[1]) for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
#    valid_neighbors = []
#    for neighbor in neighbors:
#        if in_bounds(grid, neighbor) and is_traversable(grid, neighbor):
#            wall_adjacent = False
#            for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
#                wall_node = (neighbor[0] + d[0], neighbor[1] + d[1])
#                if in_bounds(grid, wall_node) and grid[wall_node[0]][wall_node[1]] == 2:
#                    wall_adjacent = True
#                    break
#            if not wall_adjacent or grid[current[0]][current[1]] == 1:
#                valid_neighbors.append(neighbor)
#            elif wall_adjacent:
#                for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
#                    next_node = (neighbor[0] + d[0], neighbor[1] + d[1])
#                    if in_bounds(grid, next_node) and grid[next_node[0]][next_node[1]] == 1:
#                        valid_neighbors.append(neighbor)
#                        break
#    return valid_neighbors

def get_clearance(grid, node, clearance):
    for i in range(-clearance, clearance + 1):
        for j in range(-clearance, clearance + 1):
            check_node = (node[0] + i, node[1] + j)
            if in_bounds(grid, check_node) and grid[check_node[0]][check_node[1]] == 2:
                return False
    return True

def get_neighbors(grid, current, goal, clearance=5):
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

class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('turtlebot')

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

        #Publisher to visualization marker
        self.marker_pub = self.create_publisher(
            Marker,
            '/visualization_marker',
            10)
        self.marker_id = 0
        self.clean_timer = self.create_timer(0.7, self.clean_timer_callback)

        #Publisher to visualization marker array
        self.marker_array_pub = self.create_publisher(
            MarkerArray,
            'visualization_marker_array',
            10)
        self.deleteRvizMarkers()

        #Publisher to edit the turtle's speed and rotation
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10)
        self.cmd_vel = Twist()
        self.stopped = False

        #Create turn timer
        self.start_turn_angle = -1
        self.desired_turn_angle = 0
        self.turn_timer = self.create_timer(0.1, self.turn_timer_callback)

        #SLAM map
        self.mapPosition = None
        self.mapSize = (320,320)
        self.mapRes = 50
        self.mapExplored = []
        self.map = np.zeros(self.mapSize,dtype=int)

        self.search_path = None
        self.search_next_step = None
        self.search_goal = None
        #self.map_pub = self.create_publisher(
        #    OccupancyGrid,
        #    "/map",
        #    10)

        #grid_data_np = np.array(self.map)
        #self.occupancy_grid = OccupancyGrid()
        #self.occupancy_grid.header.frame_id = 'map'
        #self.occupancy_grid.info.resolution = 0.1
        #self.occupancy_grid.info.width = grid_data_np.shape[1]
        #self.occupancy_grid.info.height = grid_data_np.shape[0]
        #self.occupancy_grid.info.origin.position.x = 0.0
        #self.occupancy_grid.info.origin.position.y = 0.0
        #self.occupancy_grid.info.origin.position.z = 0.0
        #self.occupancy_grid.info.origin.orientation.x = 0.0
        #self.occupancy_grid.info.origin.orientation.y = 0.0
        #self.occupancy_grid.info.origin.orientation.z = 0.0
        #self.occupancy_grid.info.origin.orientation.w = 1.0
        #self.occupancy_grid.data = np.ravel(grid_data_np, order='C').tolist()
        #
        #self.map_pub.publish(self.occupancy_grid)


    def deleteRvizMarkers(self):
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.id = 0
        #marker.ns = 0
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        self.marker_array_pub.publish(marker_array_msg)

    def stop(self):
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(self.cmd_vel)

    def lidar_callback(self, msg):
        #print(len(msg.ranges))

        self.lidar_ranges = msg.ranges
        front_range, front_side_range = self.check_distance()

        if not self.stopped and self.start_turn_angle == -1:
            if self.position is not None:
                self.update_slam_map()

            if self.search_path is None:
                # Move the robot forward
                self.cmd_vel.linear.x = 0.5
                self.cmd_vel_pub.publish(self.cmd_vel)
                self.get_logger().info('Moving robot forward. Dist = ' + str(front_range) + " , " + str(front_side_range))

    def odometry_callback(self, msg):
        self.position = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        self.orientation = self.quaternion_to_euler(quat.x,quat.y,quat.z,quat.w)

        if abs(self.orientation.roll) >= 5:
            self.get_logger().info('FALLING OVER!! roll:' + str(self.orientation.roll))
        if abs(self.orientation.pitch) >= 5:
            self.get_logger().info('FALLING OVER!! pitch:' + str(self.orientation.pitch))

    def search_for_unexplored(self):
        self.stop()
        #print("Snake traversing")
        #found_goals = 0
        #self.search_goal = self.snail_traversal(self.mapPosition, min(self.mapSize), found_goals)
        #found_goals += 1
        #print("New goal: ", self.search_goal)
        #print("Searching path")
        #self.search_path = self.a_star(self.mapPosition, self.search_goal)
        #while self.search_path is None:
        #    self.search_goal = self.snail_traversal(self.mapPosition, min(self.mapSize), found_goals)
        #    found_goals += 1
        #    print("New goal: ", self.search_goal, found_goals)
        #    self.search_path = self.a_star(self.mapPosition, self.search_goal)
        print("Snake traversing")
        search_goals = self.snail_traversal(self.mapPosition, min(self.mapSize))
        print("Current goal: ", search_goals[0], 0, "/",len(search_goals))
        #self.search_path = a_star(self.map, self.mapPosition, search_goals[0])
        i = 20
        while self.search_path is None and i < len(search_goals):
            print("Current goal: ", search_goals[i], i,"/",len(search_goals))
            self.search_path = a_star(self.map, self.mapPosition, search_goals[i])
            i += 1

        if i == len(search_goals):
            print("Failed to find a route")
            input()
        else:
            self.search_next_step = 0

            with open("path.txt","w") as file:
                for step in self.search_path:
                    file.write(str(step[0]) + " " + str(step[1]) + "\n")
            print("Done saving")
            input()

    def snail_traversal(self, pos, radius):
        def in_bounds(x, y, grid):
            return 0 <= x < len(grid) and 0 <= y < len(grid[0])

        def next_direction(direction):
            return (direction + 1) % 4

        def move_position(x, y, direction):
            if direction == 0:
                return x, y - 1  # Move up
            elif direction == 1:
                return x + 1, y  # Move right
            elif direction == 2:
                return x, y + 1  # Move down
            elif direction == 3:
                return x - 1, y  # Move left

        found = []
        x,y = pos
        direction = 0
        steps_taken_in_current_direction = 0
        steps_to_take_in_current_direction = 1
        total_steps_taken = 0

        while total_steps_taken < (radius * 2 + 1) ** 2:
            if in_bounds(x, y, self.map):
                if self.map[x][y] == 0:
                    found.append((x,y))

            x, y = move_position(x, y, direction)
            steps_taken_in_current_direction += 1
            total_steps_taken += 1

            if steps_taken_in_current_direction == steps_to_take_in_current_direction:
                direction = next_direction(direction)
                steps_taken_in_current_direction = 0

                if direction == 1 or direction == 3:  # When changing to horizontal direction
                    steps_to_take_in_current_direction += 1
        return found


    #def a_star_is_valid_position(self, grid, x, y, clearance=1):
    #    height = len(grid)
    #    width = len(grid[0])
    #
    #    if x < clearance or x >= width - clearance or y < clearance or y >= height - clearance:
    #        return False
    #
    #    for i in range(-clearance, clearance + 1):
    #        for j in range(-clearance, clearance + 1):
    #            if grid[y + i][x + j] == 2:
    #                return False
    #
    #    return True
    #
    #def a_star_heuristic(self, a, b):
    #    return abs(a.x - b.x) + abs(a.y - b.y)
    #
    #def a_star(self, start, end):
    #    start_node = AStarNode(*start)
    #    end_node = AStarNode(*end)
    #
    #    open_list = []
    #    closed_list = []
    #
    #    heapq.heappush(open_list, start_node)
    #
    #    while open_list:
    #        if len(open_list) % 100 == 0:
    #            print(len(open_list))
    #        current_node = heapq.heappop(open_list)
    #
    #        closed_list.append(current_node)
    #
    #        if current_node == end_node:
    #            path = []
    #            while current_node is not None:
    #                path.append((current_node.x, current_node.y))
    #                current_node = current_node.parent
    #            return path[::-1]
    #
    #        neighbors = [
    #            AStarNode(current_node.x + dx, current_node.y + dy, current_node)
    #            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))
    #        ]
    #
    #        for neighbor in neighbors:
    #            if not self.a_star_is_valid_position(self.map, neighbor.x, neighbor.y) or neighbor in closed_list:
    #                continue
    #
    #            tentative_g = current_node.g + 1
    #            if neighbor not in open_list:
    #                heapq.heappush(open_list, neighbor)
    #            elif tentative_g >= neighbor.g:
    #                continue
    #
    #            neighbor.parent = current_node
    #            neighbor.g = tentative_g
    #            neighbor.h = self.a_star_heuristic(neighbor, end_node)
    #            neighbor.f = neighbor.g + neighbor.h
    #
    #    return None

    #Bresenham's line algorithm
    def bresenham(self,start,end):

        x0, y0 = start
        x1, y1 = end
        """Yield integer coordinates on the line from (x0, y0) to (x1, y1).

        Input coordinates should be integers.

        The result will contain both the start and the end point.
        """
        dx = x1 - x0
        dy = y1 - y0

        xsign = 1 if dx > 0 else -1
        ysign = 1 if dy > 0 else -1

        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        D = 2*dy - dx
        y = 0

        coords = []
        for x in range(dx + 1):
            coords.append((( x0 + x*xx + y*yx),( y0 + x*xy + y*yy)))
            if D >= 0:
                y += 1
                D -= 2*dx
            D += 2*dy
        return coords

    def update_slam_map(self):
        positionCount = 0

        px = len(self.map)//2 + int(round(self.position.x,5) * self.mapRes)
        py = len(self.map[0])//2 + int(round(self.position.y,5) * self.mapRes)
        self.mapPosition = (px,py)

        for deg in range(len(self.lidar_ranges)):
            if math.isinf(self.lidar_ranges[deg]):
                continue

            mark = Point()
            x = self.lidar_ranges[deg] * math.cos(math.radians(deg + self.orientation.yaw))
            y = self.lidar_ranges[deg] * math.sin(math.radians(deg + self.orientation.yaw))
            mark.x = self.position.x + x
            mark.y = self.position.y + y

            nx = len(self.map)//2 + int(round(self.position.x,5) * self.mapRes) + int(round(x,5) * self.mapRes)
            ny = len(self.map[1])//2 + int(round(self.position.y,5) * self.mapRes) + int(round(y,5) * self.mapRes)

            #print(deg, nx, len(self.map)//2, self.position.x, x)
            if nx < 0:# or nx >= len(self.map):
                continue
            if ny < 0:# or ny >= len(self.map[0]):
                continue


            if self.map[nx][ny] == 0:
                self.map[nx][ny] = 2
                positionCount += 1

            points = self.bresenham((px,py),(nx,ny))

            for p in points:
                try:
                    if self.map[p[0]][p[1]] == 0 and p[0] != nx and p[1] != ny:
                        self.map[p[0]][p[1]] = 1
                        pPos = Pos(p[0],p[1])
                        self.mapExplored.append(pPos)
                        positionCount += 1
                except Exception as e:
                    print(str(e) + "\n" + str(p))
                    input()


        #if positionCount >= 10:
        #    print("New markers:",positionCount)
        #
        #    self.grid_pub.publish(self.grid_cells)
        #
        #    #grid_data_np = np.array(self.map)
        #    #self.occupancy_grid.data = np.ravel(grid_data_np, order='C').tolist()
        #    #self.map_pub.publish(self.occupancy_grid)
        #
        #    #self.place_markers(positions)
        print("New markers:",positionCount)
        if positionCount <= 30 and self.search_next_step is None:
            with open("map.txt","w") as file:
                a = 3
                b = 4
                self.map[57][40],a = a,self.map[57][40]
                self.map[64][46],b = b,self.map[64][46]
                for x in range(len(self.map)):
                    for y in range(len(self.map[0])):
                        file.write(str(self.map[x][y]) + " ")
                    file.write("\n")
                self.map[57][40],a = a,self.map[57][40]
                self.map[64][46],b = b,self.map[64][46]
            print("Done saving")
            self.search_for_unexplored()
            input()

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
            self.start_turn(self.get_random_dir())
        else:
            self.stopped = False

        return front_range, front_side_range

    def clean_timer_callback(self):
        if self.start_turn_angle == -1 and self.position is not None:
            self.place_marker()

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
        if self.orientation is None:
            return

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
        dir = random.randint(90,140)
        if random.randint(0,1) == 0:
            dir *= -1
        print(dir)
        return dir

    def place_marker(self):
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

        marker.pose.position = self.position
        #print(self.position)
        self.marker_pub.publish(marker)

    def place_marker_at_pos(self, pos):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.1
        marker.color.a = 0.7
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        marker.pose.position = pos
        #print(self.position)
        self.marker_pub.publish(marker)

    def place_markers(self, positions):
        markers = []
        for pos in positions:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = self.marker_id
            self.marker_id += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1
            marker.color.a = 0.7
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.pose.position = pos
            markers.append(marker)

        marker_array = MarkerArray()
        marker_array.markers = markers
        self.marker_array_pub.publish(marker_array)


def bresenham(start,end):
    x1, y1 = start
    x2, y2 = end

    x,y = x1,y1
    dx = abs(x2 - x1)
    dy = abs(y2 -y1)

    if dx == 0:
        return []
    gradient = dy/float(dx)

    if gradient > 1:
        dx, dy = dy, dx
        x, y = y, x
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    p = 2*dy - dx
    # Initialize the plotting points
    coords = [(x,y)]

    for k in range(2, dx + 2):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy

        x = x + 1 if x < x2 else x - 1

        coords.append((x,y))
    return coords

def bresenham2(start,end):
    x1, y1 = start
    x2, y2 = end

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if (dx <= dy):
        x1,y1 = y1,x2
        x2,y2 = y2,x2
        dx,dy = dy,dx

    pk = 2 * dy - dx

    coords = []
    for i in range(0,dx+1):
        coords.append((x1,y1))

        if(x1<x2):
            x1 = x1 + 1
        else:
            x1 = x1 - 1
        if (pk < 0):
            pk = pk + 2 * dy
        else:
            if(y1<y2):
                y1 = y1 + 1
            else:
                y1 = y1 - 1

            pk = pk + 2 * dy - 2 * dx
    return coords

def bresenham3(start,end):

    x0, y0 = start
    x1, y1 = end
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).

    Input coordinates should be integers.

    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    coords = []
    for x in range(dx + 1):
        coords.append((( x0 + x*xx + y*yx),( y0 + x*xy + y*yy)))
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy
    return coords

def main(args=None):
    #print("(10,5)" , bresenham3((5,5),(10,5)))
    #print("(10,10)" ,bresenham3((5,5),(10,10)))
    #print("(5,10)" , bresenham3((5,5),(5,10)))
    #print("(0,10)" , bresenham3((5,5),(0,10)))
    #print("(0,5)" ,  bresenham3((5,5),(0,5)))
    #print("(0,0)" ,  bresenham3((5,5),(0,0)))
    #print("(5,0)" ,  bresenham3((5,5),(5,0)))
    #print("(10,0)" , bresenham3((5,5),(10,0)))
    #
    #start_time = time.monotonic()
    #bresenham3((175,175),(175,0))
    #end_time = time.monotonic()  # Record end time
    #elapsed_ms = (end_time - start_time) * 1000  # Calculate elapsed milliseconds
    #print(f"Elapsed time: {elapsed_ms:.2f} ms")
    #
    #return

    rclpy.init(args=args)

    marker_reset = rclpy.create_node('marker_reset')
    marker_pub = marker_reset.create_publisher(MarkerArray, '/visualization_marker_array', 10)
    marker_array = MarkerArray()
    marker_pub.publish(marker_array)

    turtlebot = LidarSubscriber()

    client = turtlebot.create_client(Empty, '/reset_simulation')
    request = Empty.Request()
    future = client.call_async(request)

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        turtlebot.stop()

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
