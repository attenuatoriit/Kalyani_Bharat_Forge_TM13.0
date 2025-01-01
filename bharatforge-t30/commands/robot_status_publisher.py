import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yaml
import re  

CONFIG_PATH = "/../rosslam_multi/src/m-explore-ros2/map_merge/config/params.yaml"

class RobotStatusPublisher(Node):

    def __init__(self, n):
        super().__init__('robot_status_publisher')

        # Number of robots (this can be loaded from a config file or passed directly)
        self.n = n

        # Robot names and corresponding initial status (True = free, False = not free)
        self.robot_names = [f'robot{i}' for i in range(1, n + 1)]
        self.robot_status = {robot_name: True for robot_name in self.robot_names}  # Initialize all robots as free (True)

        # Create a publisher for robot statuses
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)

        # Create a subscriber to listen to the closest robot selection topic
        self.closest_robot_subscriber = self.create_subscription(
            String,
            'closest_robot',
            self.closest_robot_callback,
            10
        )

        # Create a subscriber to listen to the robot_reached_goal topic
        self.robot_reached_goal_subscriber = self.create_subscription(
            String,
            'robot_reached_goal',
            self.robot_reached_goal_callback,
            10
        )

        # Create a timer to periodically check and publish the robot status (every 2 seconds)
        self.create_timer(2.0, self.publish_robot_status)

    def publish_robot_status(self):
        """Publish the status of the robots."""
        status_msg = String()

        # Format the robot names with their corresponding status (True = free, False = not free)
        status_list = [f"{robot_name}: {'True' if status else 'False'}" 
                       for robot_name, status in self.robot_status.items()]

        # Combine all status information into a single string
        status_msg.data = ", ".join(status_list)

        # Publish the status
        self.status_publisher.publish(status_msg)
        self.get_logger().info(f"Published status: {status_msg.data}")

    def closest_robot_callback(self, msg):
        """Callback function to update robot status when a closest robot is selected."""
        closest_robot_info = msg.data
        # The message should be in the format: "robot_name, Goal: x=..., y=..."
        robot_name = closest_robot_info.split(",")[0].strip()  # Extract the robot name from the message

        # Change the status of the closest robot to False (not free)
        if robot_name in self.robot_status:
            self.robot_status[robot_name] = False
            self.get_logger().info(f"Robot {robot_name} marked as not free.")
        else:
            self.get_logger().error(f"Received status for unknown robot: '{robot_name}'.")

    def robot_reached_goal_callback(self, msg):
        """Callback function to update robot status when a robot reaches the goal."""
        received_data = msg.data.strip()
        # Use regex to extract the robot name at the beginning of the message
        match = re.match(r'^(\w+)', received_data)
        if match:
            robot_name = match.group(1)
            if robot_name in self.robot_status:
                self.robot_status[robot_name] = True
                self.get_logger().info(f"Robot {robot_name} marked as free (reached goal).")
            else:
                self.get_logger().error(f"Received status for unknown robot: '{robot_name}'.")
        else:
            self.get_logger().error(f"Failed to parse robot name from message: '{received_data}'.")


def main(args=None):
    rclpy.init(args=args)

    # Path to the params.yaml file (you can adjust the path as needed)
    config_path = CONFIG_PATH

    # Load number_of_robots from the YAML file
    try:
        with open(config_path, "r") as file:
            params = yaml.safe_load(file)
            number_of_robots = params["map_merge"]["ros__parameters"]["number_of_robots"]
    except Exception as e:
        print(f"Error reading params.yaml: {e}")
        number_of_robots = 2  

    # Start the node with the loaded number of robots
    node = RobotStatusPublisher(n=number_of_robots)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

