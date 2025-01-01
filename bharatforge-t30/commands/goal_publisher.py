import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String
import threading
import ast  # To safely evaluate the string as a Python literal


class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')

        # Publisher for the goal_pose topic using PoseArray
        self.publisher = self.create_publisher(PoseArray, 'goal_pose', 10)

        # Subscribe to the object_history topic (which will send a string with the dictionary of locations)
        self.subscription = self.create_subscription(
            String,
            'object_history',
            self.object_history_callback,
            10
        )

        # Store the locations received from the object_history topic
        self.locations = {}

        # Start the input loop in a separate thread
        input_thread = threading.Thread(target=self.input_loop, daemon=True)
        input_thread.start()

    def object_history_callback(self, msg: String):
        """Callback function to process object history data."""
        try:
            # Convert the string message to a dictionary safely
            new_locations = ast.literal_eval(msg.data)
            if isinstance(new_locations, dict):
                self.locations = new_locations
                self.get_logger().info(f"Updated locations: {self.locations}")
            else:
                self.get_logger().error("Received data is not a valid dictionary.")
        except (ValueError, SyntaxError) as e:
            self.get_logger().error(f"Failed to parse object history data: {e}")

    def input_loop(self):
        while rclpy.ok():
            try:
                # Get user input synchronously
                item = input("\nEnter the name of the location (or 'exit' to quit): ").strip().lower()

                if item == "exit":
                    self.get_logger().info("Exiting goal publisher...")
                    rclpy.shutdown()
                    return

                if item in self.locations:
                    urgency = input("Enter urgency level (0 for normal, 1 for high): ").strip()
                    if urgency not in ["0", "1"]:
                        self.get_logger().error("Invalid urgency level. Please enter 0 or 1.")
                        continue

                    self.publish_goals(item, int(urgency))
                else:
                    self.get_logger().error(f"Invalid location: {item}. Please try again.")
            except EOFError:
                # Handle end-of-file (e.g., Ctrl+D)
                self.get_logger().info("EOF received. Exiting goal publisher...")
                rclpy.shutdown()
                return
            except Exception as e:
                self.get_logger().error(f"An error occurred: {e}")
                rclpy.shutdown()
                return

    def publish_goals(self, item_name, urgency):
        """Publish goals for a specific item based on the received locations."""
        coordinates = self.locations[item_name]
        self.get_logger().info(f"Publishing goals for {item_name} with urgency {urgency}:")

        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'  # Ensure the frame matches your setup
        pose_array.header.stamp = self.get_clock().now().to_msg()

        for idx, coord in enumerate(coordinates, start=1):
            pose = Pose()
            pose.position.x = float(coord[0])
            pose.position.y = float(coord[1])
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0

            # Add the pose to the PoseArray (no need to add a header to each individual pose)
            pose_array.poses.append(pose)

            self.get_logger().info(f"  {idx}. x={coord[0]}, y={coord[1]}")

        self.publisher.publish(pose_array)
        self.get_logger().info(f"Published PoseArray with {len(coordinates)} poses for '{item_name}' and urgency {urgency}.")

    def destroy_node(self):
        super().destroy_node()
        self.get_logger().info("GoalPublisher node destroyed.")


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('GoalPublisher Node Stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

