#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
# from tf_transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
import subprocess
import math
import threading
import time
from std_msgs.msg import String
import json

NUM_BOTS = 4

class SingleBotNavigator(Node):
    def __init__(self, robot_namespace):
        super().__init__('single_bot_navigator_' + robot_namespace.strip('/'))

        # Robot namespace
        self.robot_namespace = robot_namespace

        # Action server name+namespace
        if self.robot_namespace:
            self.action_server_name = f'{self.robot_namespace}/navigate_to_pose'
        else:
            self.action_server_name = 'navigate_to_pose'
        
        self.get_logger().info(f'Using action server: {self.action_server_name}')

    def send_goal(self, x, y, theta):
        # Convert theta from degrees to radians
        theta_rad = theta * (3.14159265 / 180.0)
        q = quaternion_from_euler(0, 0, theta_rad)

        # Construct the command
        command = [
            'ros2', 'action', 'send_goal', self.action_server_name, 'nav2_msgs/action/NavigateToPose',
            f'{{pose: {{header: {{frame_id: "map"}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{x: {q[0]}, y: {q[1]}, z: {q[2]}, w: {q[3]}}}}}}}}}'
        ]

        self.get_logger().info(f'Sending goal to navigate to the specified pose: x={x}, y={y}, theta={theta} degrees')

        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            self.get_logger().info('Goal sent successfully.')
            self.get_logger().info(result.stdout)
        else:
            self.get_logger().error('Failed to send goal.')
            self.get_logger().error(result.stderr)

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    """
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

def euler_from_quaternion(quaternion):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    :param quaternion: (x, y, z, w) quaternion
    :returns: (roll, pitch, yaw) euler angles in radians
    """
    x, y, z, w = quaternion

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(sinp) if abs(sinp) <= 1.0 else math.copysign(math.pi / 2, sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class MultiRobotCollisionAvoidance(Node):
    def __init__(self, num_robots=4, threshold_radius=1):
        super().__init__('multi_robot_collision_avoidance')

        # Parameters
        self.num_robots = num_robots
        self.threshold_radius = threshold_radius
        self.collision_cooldown = False

        # Data storage for odometry
        self.odom_data = {}
        self.laser_scan_storage = {}
        self.goal_publishers = {}

        # Subscribers for odometry and laser scan
        self.odom_subscribers = []
        self.laser_scan_subscribers = []
        self.robot_navigators = {}
        self.collision_cooldown = False
        self.collision_status_pub = self.create_publisher(String, 'collision_status', 10)
        # Create subscriptions for each robot
        for i in range(1, self.num_robots + 1):
            robot_name = f'robot{i}'
            odom_topic = f'/{robot_name}/odom'
            scan_topic = f'/{robot_name}/scan'
            self.get_logger().info(f'Subscribing to {odom_topic} and {scan_topic}')

            # Initialize odometry data for the robot
            self.odom_data[robot_name] = {'received': False, 'position': (0.0, 0.0)}
            self.robot_navigators[robot_name] = SingleBotNavigator(robot_namespace=f'/{robot_name}')
            # Subscribe to odometry
            self.odom_subscribers.append(
                self.create_subscription(
                    Odometry,
                    odom_topic,
                    lambda msg, robot_name=robot_name: self.update_odom(robot_name, msg),
                    10
                )
            )

            # Subscribe to laser scan
            self.laser_scan_subscribers.append(
                self.create_subscription(
                    LaserScan,
                    scan_topic,
                    lambda msg, robot_name=robot_name: self.laser_scan_data(robot_name, msg),
                    10
                )
            )

    def reset_collision_cooldown(self):
        self.collision_cooldown = False

    def update_odom(self, robot_name, msg):
        """
        Updates odometry data for a robot, including position and yaw.
        """
        self.odom_data[robot_name]['received'] = True
        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
    
        # Extract quaternion values from the odometry message
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
    
        # Convert quaternion to Euler angles (yaw, pitch, roll)
        _, _, yaw = euler_from_quaternion(quaternion)

        # Update odometry data with both position and yaw
        self.odom_data[robot_name]['position'] = position
        self.odom_data[robot_name]['yaw'] = yaw

        # Call the collision check function
        if not self.collision_cooldown:
            self.check_collisions()

    def laser_scan_data(self, robot_name, msg):
        """
        Store laser scan data
        """
        if not msg.ranges:
            self.get_logger().warn(f'[{robot_name}] Received empty LaserScan ranges.')
            return

        self.laser_scan_storage[robot_name] = {'ranges': msg.ranges, 'angle_min': msg.angle_min}

    def check_collisions(self):
        """
        Checks if any two robots are within the threshold radius and adjusts their goals.
        """
        robots = list(self.odom_data.keys())
        collision_dict = {robot: False for robot in robots}
        if self.collision_cooldown:
        # Initialize collision status for each robot
            for i in range(len(robots)):
                for j in range(i + 1, len(robots)):
                    robot1 = robots[i]
                    robot2 = robots[j]
                
                    if self.odom_data[robot1]['received'] and self.odom_data[robot2]['received']:
                        pos1 = self.odom_data[robot1]['position']
                        pos2 = self.odom_data[robot2]['position']

                        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

                        if distance < self.threshold_radius and not self.collision_cooldown:
                            self.get_logger().warn(f'{robot1} and {robot2} are within {distance:.2f}m (threshold: {self.threshold_radius}m)')
                            
                            # Update collision status for both robots
                            collision_dict[robot1] = True
                            collision_dict[robot2] = True
                            
                            new_goal_robot1 = self.calculate_new_goal(robot1)
                            new_goal_robot2 = self.calculate_new_goal(robot2)
                            theta = 0
                            thread1 = threading.Thread(target=self.robot_navigators[robot1].send_goal, args=(new_goal_robot1[0], new_goal_robot1[1], theta))
                            thread2 = threading.Thread(target=self.robot_navigators[robot2].send_goal, args=(new_goal_robot2[0], new_goal_robot2[1], theta))
                            thread1.start()
                            thread2.start()

                            self.collision_cooldown = True
                            self.create_timer(5.0, self.reset_collision_cooldown)
                            #time.sleep(5)
        # Publish the collision dictionary as a JSON string
        status_msg = String()
        status_msg.data = json.dumps(collision_dict)
        self.collision_status_pub.publish(status_msg)

    def calculate_new_goal(self, robot_name):
        """
        Calculates a new goal position.
        """
        laser_scan = self.laser_scan_storage.get(robot_name)
        current_position = self.odom_data[robot_name]['position']

        if laser_scan and laser_scan['ranges']:
            max_distance = max(laser_scan['ranges'])
            angle_of_max_clear_distance = laser_scan['angle_min'] + laser_scan['ranges'].index(max_distance) * 0.1

            d = 2.0  # desired distance to move
            k = 0.5  # minimum distance from obstacle

            # Calculate the distance to the nearest obstacle along the angle
            nearest_obstacle_distance = min(
                [rng for rng, angle in zip(laser_scan['ranges'], 
                [laser_scan['angle_min'] + i * 0.1 for i in range(len(laser_scan['ranges']))]) 
                if abs(angle - angle_of_max_clear_distance) < 0.05],
                default=max_distance
            )

            # Ensure the new goal is at least k distance away from the nearest obstacle
            move_distance = min(d, nearest_obstacle_distance - k) if nearest_obstacle_distance > k else 0.0

            new_x = current_position[0] + move_distance * math.cos(angle_of_max_clear_distance)
            new_y = current_position[1] + move_distance * math.sin(angle_of_max_clear_distance)

            return new_x, new_y
        return current_position


def main(args=None):
    rclpy.init(args=args)

    # Create a multi-robot collision avoidance node with n robots
    collision_avoidance = MultiRobotCollisionAvoidance(num_robots=NUM_BOTS)

    # Spin the node to keep it alive
    rclpy.spin(collision_avoidance)

    # Shutdown ROS2 client library
    collision_avoidance.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
