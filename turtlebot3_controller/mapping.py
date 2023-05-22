import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Int32MultiArray
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker

import math
import numpy as np

class Euler:
    def __init__(self, roll, pitch, yaw):
        self.roll = math.degrees(roll)
        self.pitch = math.degrees(pitch)
        self.yaw = math.degrees(yaw)

    def __str__(self):
        return "roll: " + str(round(self.roll,2)) + ", pitch: " + str(round(self.pitch,2)) + ", yaw: " + str(round(self.yaw,2))

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

class MappingNode(Node):

    def __init__(self):
        super().__init__('mapping')

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

        self.map_pub = self.create_publisher(
            Int32MultiArray,
            "/map_data",
            10
        )

        #SLAM map
        self.mapUpdated = False
        self.mapPosition = None
        self.mapSize = (320,320)
        self.mapRes = 50
        self.mapExplored = []
        self.mapLimits = DirLimit(320,0,320,0)
        self.map = np.zeros(self.mapSize,dtype=int)

        self.map_occu_pub = self.create_publisher(
            OccupancyGrid,
            "/map_occu",
            10)

        grid_data_np = np.array(self.occu_map_convert())
        self.occupancy_grid = OccupancyGrid()
        self.occupancy_grid.header.frame_id = 'map'
        self.occupancy_grid.info.resolution = 0.022
        self.occupancy_grid.info.width = grid_data_np.shape[1]
        self.occupancy_grid.info.height = grid_data_np.shape[0]
        self.occupancy_grid.info.origin.position.x = 3.4782 #verical, bigger down, smaller up
        self.occupancy_grid.info.origin.position.y = -3.55 #horizontal, smaller left, bigger right
        self.occupancy_grid.info.origin.position.z = 0.0
        rotation_angle_rad = math.radians(90) / 2  # Converting to radians and dividing by 2 for quaternion
        self.occupancy_grid.info.origin.orientation.x = 0.0
        self.occupancy_grid.info.origin.orientation.y = 0.0
        self.occupancy_grid.info.origin.orientation.z = math.sin(rotation_angle_rad)
        self.occupancy_grid.info.origin.orientation.w = math.cos(rotation_angle_rad)
        self.occupancy_grid.data = np.ravel(grid_data_np, order='C').tolist()
        
        self.map_occu_pub.publish(self.occupancy_grid)

        #Publisher to visualization marker
        self.marker_pub = self.create_publisher(
            Marker,
            '/pos_marker',
            10)
        self.pos_marker_id = 0
        self.marker_id = 0
        self.clean_timer = self.create_timer(0.1, self.clean_timer_callback)

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges

        if self.position is not None:
            self.update_slam_map()

            if self.mapUpdated:
                #publish map and limits
                msg = Int32MultiArray()
                msg.data = self.mapLimits.getAsArray() + [int(item) for sublist in self.map for item in sublist]
                self.map_pub.publish(msg)

                #publish occupancy map
                grid_data_np = np.array(self.occu_map_convert())
                self.occupancy_grid.data = np.ravel(grid_data_np, order='C').tolist()
                self.map_occu_pub.publish(self.occupancy_grid)   

                ##publish mapped cells to rviz2
                #self.grid_pub.publish(self.grid_cells)                             
                #self.get_logger().info("Map data published")

    def occu_map_convert(self):
        new_matrix = [[0] * len(self.map[0]) for _ in range(len(self.map))]  # Create a new matrix with the same dimensions

        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if self.map[i][j] == 1:
                    new_matrix[319-i][j] = 50
                elif self.map[i][j] == 2:
                    new_matrix[319-i][j] = 100
                else:
                    new_matrix[319-i][j] = self.map[i][j]  # Copy unchanged values

        return new_matrix
    
    def clean_timer_callback(self):
        if self.position is not None:
            self.place_marker()
    
    def place_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = self.pos_marker_id
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

        px = self.mapSize[0]//2 + int(round(self.position.x,5) * self.mapRes)
        py = self.mapSize[1]//2 + int(round(self.position.y,5) * self.mapRes)
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

            if nx < 0:# or nx >= len(self.map):
                continue
            if ny < 0:# or ny >= len(self.map[0]):
                continue


            if self.map[nx][ny] == 0:
                self.mapUpdated = True
                self.map[nx][ny] = 2

                if ny < self.mapLimits.up:
                    self.mapLimits.up = ny
                if ny > self.mapLimits.down:
                    self.mapLimits.down = ny
                if nx < self.mapLimits.left:
                    self.mapLimits.left = nx
                if nx > self.mapLimits.right:
                    self.mapLimits.right = nx

                positionCount += 1

            points = self.bresenham((px,py),(nx,ny))

            for p in points:
                try:
                    if self.map[p[0]][p[1]] == 0 and p[0] != nx and p[1] != ny:
                        self.mapUpdated = True
                        self.map[p[0]][p[1]] = 1
                        pPos = Pos(p[0],p[1])
                        self.mapExplored.append(pPos)
                        positionCount += 1
                except Exception as e:
                    print(str(e) + "\n" + str(p))


def main(args=None):
    rclpy.init(args=args)

    turtlebot = MappingNode()

    try:
        rclpy.spin(turtlebot)
    except KeyboardInterrupt:
        pass

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()