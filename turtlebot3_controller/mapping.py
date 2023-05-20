import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Int32MultiArray

import math
import numpy as np

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
        super().__init__('mappingnode')

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

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges

        if self.position is not None:
            self.update_slam_map()

            if self.mapUpdated:
                #publish map and limits
                msg = Int32MultiArray()
                msg.data = self.mapLimits.getAsArray() + [item for sublist in self.map for item in sublist]
                self.map_pub.publish(msg)
                self.get_logger().info("Map data published")

    def odometry_callback(self, msg):
        self.position = msg.pose.pose.position

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
        turtlebot.stop()

    turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()