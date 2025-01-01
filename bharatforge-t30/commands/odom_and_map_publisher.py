import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import String
import numpy as np
import os
import yaml

MAP_SIZE = 1250

class OdomAndMapPublisher(Node):
    def __init__(self):
        super().__init__('odom_and_map_publisher')

        # Subscriptions to odometry and map topics for 8 robots
        self.sub_odom = [
            self.create_subscription(Odometry, f'/robot{i+1}/odom', self.odom_callback(i), 10)
            for i in range(8)
        ]
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Subscription for detected objects
        self.sub_objects = self.create_subscription(String, '/detected_objects', self.objects_callback, 10)

        # Publishers for the updated map and object history
        self.pub_updated_map = self.create_publisher(OccupancyGrid, '/updated_map', 10)
        self.pub_object_history = self.create_publisher(String, '/object_history', 10)

        # Timer to publish synchronized data at 1 Hz
        self.timer = self.create_timer(1.0, self.publish_synced_data)

        # Variables to store the latest received data
        self.robot_odom = [None] * 8
        self.map_data = None
        self.object_positions = {}  # Stores the latest position of each object
        self.object_history = {}    # Dictionary of object name to a set of coordinates
        self.object_index_mapping = self.load_object_index_mapping()

    def load_object_index_mapping(self):
        """Load object index mapping from yolov3.txt."""
        mapping = {}
        file_path = os.path.join(os.getcwd(), "yolov3.txt")
        try:
            with open(file_path, "r") as file:
                for idx, line in enumerate(file):
                    obj_name = line.strip()
                    if obj_name:
                        mapping[obj_name] = 99 - idx 
        except FileNotFoundError:
            self.get_logger().error(f"'yolov3.txt' not found at {file_path}.")
        return mapping

    def odom_callback(self, idx):
        """Generate a callback for a specific robot odometry."""
        def callback(msg):
            self.robot_odom[idx] = msg
        return callback

    def map_callback(self, msg):
        self.map_data = msg

    def objects_callback(self, msg):
        """Process detected objects and store their coordinates."""
        data = msg.data
        object_entries = data.split("\n")

        for entry in object_entries:
            if entry.strip():
                try:
                    object_name, x, y, z = self.parse_object_entry(entry)
                    if object_name in self.object_index_mapping:
                        new_position = (x, y)

                        if object_name not in self.object_positions:
                            self.object_positions[object_name] = set()

                        if not self.is_position_within_threshold(object_name, new_position):
                            self.object_positions[object_name].add(new_position)

                            if object_name not in self.object_history:
                                self.object_history[object_name] = set()
                            self.object_history[object_name].add(new_position)
                except (ValueError, IndexError) as e:
                    self.get_logger().warning(f"Failed to parse entry: {entry}, error: {e}")

    def parse_object_entry(self, entry):
        """Parse an object entry to extract the object name and coordinates."""
        object_start = entry.find('Object: ') + len('Object: ')
        object_end = entry.find(',', object_start)
        object_name = entry[object_start:object_end].strip()

        x_start = entry.find('x=') + len('x=')
        x_end = entry.find(',', x_start)
        x = float(entry[x_start:x_end].strip())

        y_start = entry.find('y=') + len('y=')
        y_end = entry.find(',', y_start)
        y = float(entry[y_start:y_end].strip())

        z_start = entry.find('z=') + len('z=')
        z_end = entry.find(',', z_start)
        z = float(entry[z_start:z_end].strip())

        return object_name, x, y, z

    def is_position_within_threshold(self, object_name, new_position, threshold=1.0):
        """Check if the new position is within a threshold distance."""
        if object_name in self.object_positions:
            for (x, y) in self.object_positions[object_name]:
                if abs(x - new_position[0]) < threshold and abs(y - new_position[1]) < threshold:
                    return True
        return False

    def update_map_with_robots_and_objects(self, map_matrix, robot_positions, object_positions, resolution, origin):
        """Update the map grid with robot and object positions."""
        map_height, map_width = map_matrix.shape
        updated_map = map_matrix.copy()

        for idx, robot_pos in enumerate(robot_positions):
            if robot_pos is not None:
                x_idx = int((robot_pos.x - origin.x) / resolution)
                y_idx = int((robot_pos.y - origin.y) / resolution)
                if 0 <= x_idx < map_width and 0 <= y_idx < map_height:
                    updated_map[y_idx, x_idx] = 99 - idx

        for obj_name, positions in object_positions.items():
            if obj_name in self.object_index_mapping:
                for (x, y) in positions:
                    x_idx = int((x - origin.x) / resolution)
                    y_idx = int((y - origin.y) / resolution)
                    if 0 <= x_idx < map_width and 0 <= y_idx < map_height:
                        updated_map[y_idx, x_idx] = self.object_index_mapping[obj_name]
        return updated_map

    def pad_map_to_500x500(self, map_matrix):
        """Pad the map matrix symmetrically with -1 to make it 500x500."""
        current_height, current_width = map_matrix.shape
        target_height, target_width = MAP_SIZE, MAP_SIZE
        pad_height = target_height - current_height
        pad_width = target_width - current_width

        if pad_height < 0 or pad_width < 0:
            raise ValueError("Current map is larger than the target size of 900x900.")

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_map = np.pad(
            map_matrix,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=-1,
        )
        return padded_map

    def publish_synced_data(self):
        """Publish synchronized map data with robot and object positions."""
        if self.map_data:
            map_width = self.map_data.info.width
            map_height = self.map_data.info.height
            map_resolution = self.map_data.info.resolution
            map_origin = self.map_data.info.origin.position
            map_matrix = np.array(self.map_data.data).reshape((map_height, map_width))

            robot_positions = [
                odom.pose.pose.position if odom else None for odom in self.robot_odom
            ]

            updated_map = self.update_map_with_robots_and_objects(
                map_matrix, robot_positions, self.object_positions, map_resolution, map_origin
            )
            updated_map = self.pad_map_to_500x500(updated_map)

            updated_map_msg = OccupancyGrid()
            updated_map_msg.header = self.map_data.header
            updated_map_msg.info = self.map_data.info
            updated_map_msg.data = updated_map.flatten().tolist()
            self.pub_updated_map.publish(updated_map_msg)

            object_history_str = str({obj_name: list(coords) for obj_name, coords in self.object_history.items()})
            object_history_msg = String()
            object_history_msg.data = object_history_str
            self.pub_object_history.publish(object_history_msg)

            self.get_logger().info('Published updated map and object history.')

            # Save object history to YAML
            self.save_object_history_to_yaml()

        else:
            self.get_logger().warning('Map data is missing. Cannot publish updated map.')

    def save_object_history_to_yaml(self):
        """Save object history to a YAML file."""
        file_path = os.path.join(os.getcwd(), 'object_history.yaml')
        formatted_history = {}

        for object_name, positions in self.object_history.items():
            formatted_history[object_name] = []
            for (x, y) in positions:
                formatted_history[object_name].append({'x': x, 'y': y})

        try:
            with open(file_path, 'w') as file:
                yaml.dump(formatted_history, file, default_flow_style=False)
            self.get_logger().info(f"Object history saved to {file_path}.")
        except Exception as e:
            self.get_logger().error(f"Failed to save object history to YAML: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OdomAndMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
