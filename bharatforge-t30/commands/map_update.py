import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
import numpy as np

MAP_SIZE = 1250

class OdomAndMapPublisher(Node):
    def __init__(self):
        super().__init__('odom_and_map_publisher')
        # Subscriptions to odometry and map topics for 8 robots
        self.sub_robot1_odom = self.create_subscription(
            Odometry, '/robot1/odom', self.robot1_odom_callback, 10
        )
        self.sub_robot2_odom = self.create_subscription(
            Odometry, '/robot2/odom', self.robot2_odom_callback, 10
        )
        self.sub_robot3_odom = self.create_subscription(
            Odometry, '/robot3/odom', self.robot3_odom_callback, 10
        )
        self.sub_robot4_odom = self.create_subscription(
            Odometry, '/robot4/odom', self.robot4_odom_callback, 10
        )
        self.sub_robot5_odom = self.create_subscription(
            Odometry, '/robot5/odom', self.robot5_odom_callback, 10
        )
        self.sub_robot6_odom = self.create_subscription(
            Odometry, '/robot6/odom', self.robot6_odom_callback, 10
        )
        self.sub_robot7_odom = self.create_subscription(
            Odometry, '/robot7/odom', self.robot7_odom_callback, 10
        )
        self.sub_robot8_odom = self.create_subscription(
            Odometry, '/robot8/odom', self.robot8_odom_callback, 10
        )
        self.sub_map = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        # Publisher for the updated map
        self.pub_updated_map = self.create_publisher(OccupancyGrid, '/updated_map', 10)
        # Timer to publish synchronized data at 1 Hz
        self.timer = self.create_timer(1.0, self.publish_synced_data)
        # Variables to store the latest received data
        self.robot1_odom = None
        self.robot2_odom = None
        self.robot3_odom = None
        self.robot4_odom = None
        self.robot5_odom = None
        self.robot6_odom = None
        self.robot7_odom = None
        self.robot8_odom = None
        self.map_data = None
    def robot1_odom_callback(self, msg):
        self.robot1_odom = msg
    def robot2_odom_callback(self, msg):
        self.robot2_odom = msg
    def robot3_odom_callback(self, msg):
        self.robot3_odom = msg
    def robot4_odom_callback(self, msg):
        self.robot4_odom = msg
    def robot5_odom_callback(self, msg):
        self.robot5_odom = msg
    def robot6_odom_callback(self, msg):
        self.robot6_odom = msg
    def robot7_odom_callback(self, msg):
        self.robot7_odom = msg
    def robot8_odom_callback(self, msg):
        self.robot8_odom = msg
    def map_callback(self, msg):
        self.map_data = msg
    def update_map_with_robots(self, map_matrix, robot_positions, resolution, origin):
        """
        Update the map grid with robot positions.
        Args:
            map_matrix: 2D numpy array of map data
            robot_positions: List of robot positions (x, y)
            resolution: Map resolution
            origin: Origin position of the map
        """
        map_height, map_width = map_matrix.shape
        for idx, robot_pos in enumerate(robot_positions):
            if robot_pos is not None:
                print(f"robot_pos {idx+1}: ",robot_pos.x, robot_pos.y)
                x_idx = int((robot_pos.y) / resolution) + map_height // 2
                y_idx = int((robot_pos.x) / resolution) + map_width // 2
                # Ensure indices are within map bounds
                if(idx == 1):
                    print("##################",x_idx,y_idx)
                if 0 <= x_idx <= map_height and 0 <= y_idx <= map_width:
                    map_matrix[x_idx, y_idx] = 99 - idx  # Mark robot positions (e.g., 99, 98, ...)
        print("shape of map: ",  map_height)
        return map_matrix
    
    def pad_map_to_900x900(self, map_matrix):
        """
        Pad the map matrix symmetrically with -1 to make it 900x900.
        Args:
            map_matrix: 2D numpy array of map data
        Returns:
            Padded map matrix of size 900x900
        """
        current_height, current_width = map_matrix.shape
        target_height, target_width = MAP_SIZE, MAP_SIZE

        # Calculate the padding needed
        pad_height = target_height - current_height
        pad_width = target_width - current_width

        if pad_height < 0 or pad_width < 0:
            raise ValueError("Current map is larger than the target size of 900x900.")

        # Calculate padding for each side
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding
        padded_map = np.pad(
            map_matrix,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=-1,
        )
        return padded_map

    def publish_synced_data(self):
        """Publish synchronized map data with available robot positions."""
        if self.map_data:
            # Extract map metadata and convert to numpy array
            map_width = self.map_data.info.width
            map_height = self.map_data.info.height
            map_resolution = self.map_data.info.resolution
            map_origin = self.map_data.info.origin.position  # Map origin (x, y, z)
            map_matrix = np.array(self.map_data.data).reshape((map_height, map_width))

            updated_map_array2 = np.array(map_matrix)
            positions = np.argwhere(updated_map_array2 == 99)
            print("**************init***************", positions)

            # Collect robot positions from available odometry data
            robot_positions = [
                self.robot1_odom.pose.pose.position if self.robot1_odom else None,
                self.robot2_odom.pose.pose.position if self.robot2_odom else None,
                self.robot3_odom.pose.pose.position if self.robot3_odom else None,
                self.robot4_odom.pose.pose.position if self.robot4_odom else None,
                self.robot5_odom.pose.pose.position if self.robot5_odom else None,
                self.robot6_odom.pose.pose.position if self.robot6_odom else None,
                self.robot7_odom.pose.pose.position if self.robot7_odom else None,
                self.robot8_odom.pose.pose.position if self.robot8_odom else None,
            ]
            # Update the map with robot positions
            updated_map = self.update_map_with_robots(
                map_matrix, robot_positions, map_resolution, map_origin
            )
            updated_map = self.pad_map_to_900x900(updated_map)

            # Publish the updated map
            updated_map_msg = OccupancyGrid()
            updated_map_msg.header = self.map_data.header  # Retain original header
            updated_map_msg.info = self.map_data.info  # Retain map metadata
            updated_map_msg.data = updated_map.flatten().tolist()  # Flatten map data
            print(updated_map,updated_map.shape,np.sum(updated_map))
            self.pub_updated_map.publish(updated_map_msg)

            updated_map_array2 = np.array(updated_map_msg.data).reshape((MAP_SIZE, MAP_SIZE))

            # if(updated_map_array2.shape[0] == 900 and updated_map_array2.shape[1] == 900):
            #     np.savetxt('updated_map.txt', updated_map_array2, fmt='%d')

            self.get_logger().info('Published updated map with robot positions.')
        else:
            self.get_logger().warning('Map data is missing. Cannot publish updated map.')

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