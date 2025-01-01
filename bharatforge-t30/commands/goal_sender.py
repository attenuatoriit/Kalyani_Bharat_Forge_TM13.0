import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import threading

class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')

        # Subscriber for closest robot and goal (from ClosestRobotSelector)
        self.closest_robot_subscriber = self.create_subscription(
            String, 'closest_robot', self.closest_robot_callback, 10
        )

        # Publisher to announce when the robot reaches the goal
        self.robot_reached_goal_publisher = self.create_publisher(
            String, 'robot_reached_goal', 10
        )

        # Dictionary to store action clients for each robot
        self.navigate_clients = {}
        self.navigate_clients_lock = threading.Lock()

        # Dictionary to track ongoing navigation goals
        self.ongoing_goals = {}
        self.ongoing_goals_lock = threading.Lock()

    def closest_robot_callback(self, msg):
        """Callback function to receive the closest robot and its goal coordinates."""
        # Parse the message to get the robot name and goal coordinates
        data = msg.data.split(", Goal: ")
        if len(data) != 2:
            self.get_logger().error(f"Invalid message format: '{msg.data}'. Expected format 'robot_name, Goal: x=val, y=val'")
            return

        robot_name = data[0].strip()  # Extract robot name
        goal_coordinates = data[1].strip()

        # Split the goal coordinates into x and y values
        try:
            goal_data = goal_coordinates.split(", ")
            if len(goal_data) != 2:
                raise ValueError("Incomplete goal coordinates.")

            goal_x = float(goal_data[0].split("=")[1])
            goal_y = float(goal_data[1].split("=")[1])

            self.get_logger().info(f"Received goal for robot '{robot_name}': x={goal_x}, y={goal_y}")
        except (IndexError, ValueError) as e:
            self.get_logger().error(f"Failed to parse the goal coordinates: {e}")
            return

        # Create a PoseStamped message for the goal
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0  # No rotation (quaternion)

        # Send the closest robot to the goal in a separate thread to prevent blocking
        threading.Thread(target=self.handle_navigation, args=(robot_name, goal_pose), daemon=True).start()

    def handle_navigation(self, robot_name, goal_pose):
        """Handle sending the navigation goal to the specified robot."""
        with self.navigate_clients_lock:
            # Dynamically create an action client for the given robot if it doesn't exist
            if robot_name not in self.navigate_clients:
                self.navigate_clients[robot_name] = ActionClient(
                    self,
                    NavigateToPose,
                    f'/{robot_name}/navigate_to_pose'
                )
                self.get_logger().info(f"Created action client for '{robot_name}'.")

            client = self.navigate_clients[robot_name]

        # Wait for the action server to be available
        if not client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"NavigateToPose action server for '{robot_name}' not available after waiting.")
            return

        # Create and send the goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(f"Sending NavigateToPose goal to '{robot_name}': x={goal_pose.pose.position.x}, y={goal_pose.pose.position.y}")

        send_goal_future = client.send_goal_async(goal_msg, feedback_callback=None)
        send_goal_future.add_done_callback(lambda future: self.handle_navigation_response(robot_name, future))

    def handle_navigation_response(self, robot_name, future):
        """Handle the response from the NavigateToPose action."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"NavigateToPose goal was rejected by '{robot_name}'.")
            return

        self.get_logger().info(f"NavigateToPose goal accepted by '{robot_name}'.")

        # Track the ongoing goal
        with self.ongoing_goals_lock:
            self.ongoing_goals[robot_name] = goal_handle

        # Get the result asynchronously
        goal_handle.get_result_async().add_done_callback(
            lambda future: self.process_navigation_result(robot_name, future)
        )

    def process_navigation_result(self, robot_name, future):
        """Process the result from the NavigateToPose action."""
        try:
            # Obtain the action result
            action_result = future.result()
            result = action_result.result  # This is a boolean

            # Log the entire result for debugging purposes
            self.get_logger().debug(f"Action result for '{robot_name}': {result}")

            # Check if the navigation was successful
            if result:
                self.get_logger().info(f"'{robot_name}' successfully reached the goal!")
                self.publish_robot_reached_goal(robot_name)
            else:
                self.get_logger().error(f"'{robot_name}' failed to reach the goal.")
        except Exception as e:
            self.get_logger().error(f"Exception while processing result for '{robot_name}': {e}")
        finally:
            # Remove the goal from ongoing goals
            with self.ongoing_goals_lock:
                if robot_name in self.ongoing_goals:
                    del self.ongoing_goals[robot_name]

    def publish_robot_reached_goal(self, robot_name):
        """Publish the robot's name when it reaches the goal."""
        msg = String()
        msg.data = f"{robot_name} has reached the goal."
        self.robot_reached_goal_publisher.publish(msg)
        self.get_logger().info(f"Published: '{robot_name}' has reached the goal.")

    def destroy_node(self):
        """Clean up resources when the node is destroyed."""
        super().destroy_node()
        self.get_logger().info("GoalSender node destroyed.")

def main(args=None):
    rclpy.init(args=args)
    node = GoalSender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('GoalSender Node Stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

