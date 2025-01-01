#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
import math
import argparse

def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    return [qx, qy, qz, qw]


class MultiRobotObstacleAvoidance(Node):
    def __init__(self, num_robots, threshold_radius=0.5):
        super().__init__('multi_robot_obstacle_avoidance')

        # Initialize parameters
        self.num_robots = num_robots
        self.threshold_radius = threshold_radius
        self.robot_data = {f'robot{i}': {'odom': None, 'scan': None} for i in range(1, self.num_robots + 1)}
        self.goal_publishers = {}

        # Subscribers and publishers for each robot
        for i in range(1, self.num_robots + 1):
            robot_name = f'robot{i}'
            odom_topic = f'/{robot_name}/odom'
            scan_topic = f'/{robot_name}/scan'
            goal_topic = f'/{robot_name}/goal'

            # Subscribe to topics
            self.create_subscription(Odometry, odom_topic, lambda msg, rn=robot_name: self.update_odom(rn, msg), 10)
            self.create_subscription(LaserScan, scan_topic, lambda msg, rn=robot_name: self.update_scan(rn, msg), 10)

            # Publisher for goal
            self.goal_publishers[robot_name] = self.create_publisher(PoseStamped, goal_topic, 10)

    def update_odom(self, robot_name, msg):
        """Update odometry data for a specific robot."""
        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        _, _, yaw = self.quaternion_to_euler(quaternion)
        self.robot_data[robot_name]['odom'] = {'position': position, 'yaw': yaw}

    def update_scan(self, robot_name, msg):
        """Update laser scan data for a specific robot and check for obstacles."""
        self.robot_data[robot_name]['scan'] = msg
        self.check_obstacle(robot_name)

    def check_obstacle(self, robot_name):
        """Check if an obstacle is within the threshold radius and adjust the goal."""
        scan_data = self.robot_data[robot_name].get('scan')
        if not scan_data or not scan_data.ranges:
            return

        min_distance = min(scan_data.ranges)
        if min_distance < self.threshold_radius:
            self.get_logger().warn(f'{robot_name}: Obstacle detected within {min_distance:.2f}m.')
            new_goal = self.calculate_new_goal(robot_name)
            if new_goal:
                self.send_goal(robot_name, new_goal)

    def calculate_new_goal(self, robot_name):
        """Calculate a new goal position to avoid obstacles."""
        scan_data = self.robot_data[robot_name].get('scan')
        current_position = self.robot_data[robot_name]['odom']['position']

        if not scan_data or not scan_data.ranges:
            return None

        max_distance = max(scan_data.ranges)
        angle_max_clear = scan_data.angle_min + scan_data.ranges.index(max_distance) * scan_data.angle_increment

        new_x = current_position[0] + max_distance * math.cos(angle_max_clear)
        new_y = current_position[1] + max_distance * math.sin(angle_max_clear)

        return new_x, new_y

    def send_goal(self, robot_name, goal_position):
        """Publish a new goal for the specified robot."""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = goal_position[0]
        goal_msg.pose.position.y = goal_position[1]
        goal_msg.pose.orientation.w = 1.0  # Neutral orientation

        self.goal_publishers[robot_name].publish(goal_msg)
        self.get_logger().info(f"{robot_name}: New goal set to ({goal_position[0]:.2f}, {goal_position[1]:.2f}).")

    @staticmethod
    def quaternion_to_euler(quaternion):
        """Convert quaternion to Euler angles."""
        x, y, z, w = quaternion
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multi-Robot Obstacle Avoidance System")
    parser.add_argument('--num_robots', type=int, required=True, help="Number of robots to manage.")
    parsed_args = parser.parse_args()

    # Convert Namespace to a list of arguments for rclpy
    rclpy_args = [f'--num_robots={parsed_args.num_robots}']

    rclpy.init(args=rclpy_args)  # Pass the list of arguments
    multi_robot_avoidance = MultiRobotObstacleAvoidance(num_robots=parsed_args.num_robots)
    try:
        rclpy.spin(multi_robot_avoidance)
    except KeyboardInterrupt:
        multi_robot_avoidance.get_logger().info("Shutting down.")
    finally:
        multi_robot_avoidance.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
