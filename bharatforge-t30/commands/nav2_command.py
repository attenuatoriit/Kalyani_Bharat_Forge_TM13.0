#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import subprocess
from geometry_msgs.msg import PoseStamped

class SingleBotNavigator(Node):
    def __init__(self):
        super().__init__('single_bot_navigator')

        # Parameters
        self.declare_parameter('robot_namespace', 'robot1')  # e.g., '/robot1'
        self.robot_namespace = self.get_parameter('robot_namespace').get_parameter_value().string_value

        # Action server name+namespace
        if self.robot_namespace:
            self.action_server_name = f'{self.robot_namespace}/navigate_to_pose'
            self.goal_topic = f'{self.robot_namespace}/goal'
        else:
            self.action_server_name = 'navigate_to_pose'
            self.goal_topic = 'goal'

        self.get_logger().info(f'Using action server: {self.action_server_name}')
        self.get_logger().info(f'Subscribing to goal topic: {self.goal_topic}')

        # Subscriber for goal topic
        self.subscription = self.create_subscription(
            PoseStamped,
            self.goal_topic,
            self.goal_callback,
            10
        )

    def goal_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Extract orientation (quaternion) and convert to yaw (theta)
        import math
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        theta = math.degrees(yaw)

        self.get_logger().info(f'Received goal: x={x}, y={y}, theta={theta} degrees')
        self.send_goal(x, y, theta)

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
    import math
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)

    # Instantiate navigator nodes for each robot
    navigator_robot1 = SingleBotNavigator()
    navigator_robot1.get_logger().info("Initialized SingleBotNavigator for robot1.")

    try:
        rclpy.spin(navigator_robot1)
    except KeyboardInterrupt:
        navigator_robot1.get_logger().info('Navigation interrupted by user.')
    finally:
        navigator_robot1.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
