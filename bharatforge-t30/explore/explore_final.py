import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import subprocess
import time
import threading
from std_msgs.msg import String

# Environment parameters
MAP_SIZE = 1250
ROBOT_PIXEL_START = 99  # Robot1: 99, Robot2: 98, etc.
WAIT_TIME = 5  # Seconds to wait after reaching goals
TIMEOUT = 120
NUM_BOTS = 4

WEIGHT_PATH = '/home/TEAM30/bharatforge-t30/explore/model6.pth'


# Define the neural network model (Actor-Critic)
class ActorCritic(nn.Module):
    def __init__(self, num_inputs=4, num_bots=4):
        super(ActorCritic, self).__init__()
        self.num_bots = num_bots

        self.conv1 = nn.Conv2d(num_inputs, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((9, 9))

        self.fc = nn.Linear(64 * 9 * 9, 256)

        self.actor = nn.Linear(256, self.num_bots * 2)  # Output pred_x and pred_y for each bot
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add batch dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc(x))

        actor_output = self.actor(x)  # Direct output without activation
        value = self.critic(x)

        return actor_output, value

def quaternion_from_euler(roll, pitch, yaw):
    import math
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - \
         math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + \
         math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - \
         math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + \
         math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    return [qx, qy, qz, qw]


class ExplorationEnv(Node):
    def __init__(self, number_of_bots):
        super().__init__('exploration_env')
        self.number_of_bots = number_of_bots
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/updated_map',
            self.map_callback,
            10
        )
        self.collision_subscription = self.create_subscription(
            String,
            '/collision_status',
            self.collision_callback,
            10
        )
        self.collision_dict = {f'robot{i+1}': False for i in range(self.number_of_bots)}
        self.map_data = None
        self.map_info = None
        self.bot_positions = [None for _ in range(self.number_of_bots)]
        input_channels = 4  # Example number of input channels
        self.model = ActorCritic(num_inputs=input_channels, num_bots=self.number_of_bots)
        self.model.load_state_dict(torch.load(WEIGHT_PATH))
        self.model.eval()
        self.get_logger().info('ExplorationEnv node has been started.')

    def collision_callback(self, msg):
        import json
        try:
            collision_data = json.loads(msg.data)
            # Update collision_dict with the received data
            for robot, status in collision_data.items():
                if robot in self.collision_dict:
                    self.collision_dict[robot] = status
        except json.JSONDecodeError:
            self.get_logger().error("Invalid collision data format.")

    def map_callback(self, msg):
        try:
            msg.info.height = MAP_SIZE
            msg.info.width = MAP_SIZE
            expected_size = msg.info.height * msg.info.width
            actual_size = len(msg.data)

            if expected_size != actual_size:
                self.get_logger().error(
                    f"Map size mismatch: Expected {expected_size}, got {actual_size}."
                )
                return

            # Reshape the data
            self.map_data = np.array(msg.data, dtype=np.int8).reshape(
                msg.info.height, msg.info.width
            )
            self.map_info = msg.info
            self.process_map()  # Run the method for map processing
        except ValueError as e:
            self.get_logger().error(f"Error reshaping map data: {str(e)}")

    def process_map(self):
        if self.map_data is None:
            self.get_logger().info("Map data is None.")
            return
        if self.map_info is None:
            self.get_logger().warn('Map info is not available yet.')
            return

        # Find robot positions
        for i in range(self.number_of_bots):
            bot_pixel_value = ROBOT_PIXEL_START - i
            positions = np.argwhere(self.map_data == bot_pixel_value)
            if positions.size > 0:
                self.bot_positions[i] = positions[0]  # (y, x)
                self.get_logger().info(f"Robot {i+1} found at position: {self.bot_positions[i]}")
            else:
                self.bot_positions[i] = [0, 0]
                self.get_logger().warn(f'Robot {i+1} position not found in the map.')

        # Prepare input for the model
        state_channels = np.zeros((4, self.map_data.shape[0], self.map_data.shape[1]), dtype=np.float32)
        state_channels[0][self.map_data == -1] = 1.0  # Unexplored
        state_channels[1][self.map_data == 0] = 1.0    # Free space
        state_channels[2][self.map_data == 100] = 1.0  # Obstacles
        for idx in range(self.number_of_bots):
            state_channels[3][self.map_data == (99 - idx)] = 1.0  # Bots

        state_tensor = torch.FloatTensor(state_channels)

        with torch.no_grad():
            output, _ = self.model(state_tensor)
        output = output.view(self.number_of_bots, 2)

        target_positions = []
        free_positions = np.argwhere(self.map_data == 0)  # Positions of free explored pixels

        if len(free_positions) > 0:
            map_height, map_width = self.map_data.shape
            normalized_free_positions = free_positions / np.array([map_height - 1, map_width - 1])

            for idx in range(self.number_of_bots):
                pred_x = output[idx, 0].item() % 1.0
                pred_y = output[idx, 1].item() % 1.0

                # Compute distances to normalized free positions
                distances = np.linalg.norm(normalized_free_positions - np.array([pred_x, pred_y]), axis=1)
                nearest_idx = np.argmin(distances)
                map_x, map_y = free_positions[nearest_idx]
                target_positions.append((int(map_x), int(map_y)))
        else:
            # No free explored pixels available
            for idx in range(self.number_of_bots):
                map_x, map_y = self.bot_positions[idx]
                target_positions.append((int(map_x), int(map_y)))

        # Convert target_positions to x_pixel and y_pixel and send goals

        threads = []
        for i, (map_x, map_y) in enumerate(target_positions):
            robot_name = f'robot{i+1}'
            if not self.collision_dict.get(robot_name, False):
                y_pixel = (map_x-(MAP_SIZE // 2))*0.05
                x_pixel = (map_y-(MAP_SIZE // 2))*0.05
                robot_namespace = f'/{robot_name}'
                theta = 0

                # Create a thread to send the navigation goal
                thread = threading.Thread(target=self.send_goal, args=(robot_namespace, x_pixel, y_pixel, theta))
                threads.append(thread)
                thread.start()
            else:
                self.get_logger().info(f'Skipping goal for {robot_name} due to collision.')

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Wait additional seconds for environment to localize and map
        self.get_logger().info(f'Waiting for {WAIT_TIME} seconds...')
        time.sleep(WAIT_TIME)

    def send_goal(self, robot_namespace, x, y, theta):
        # Convert theta from degrees to radians
        theta_rad = theta * (np.pi / 180.0)
        q = quaternion_from_euler(0, 0, theta_rad)

        # Action server name with namespace
        action_server_name = f'{robot_namespace}/navigate_to_pose'

        self.get_logger().info(f'Using action server: {action_server_name}')

        # Construct the command
        command = [
            'ros2', 'action', 'send_goal', action_server_name, 'nav2_msgs/action/NavigateToPose',
            f'{{pose: {{header: {{frame_id: "map"}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{x: {q[0]}, y: {q[1]}, z: {q[2]}, w: {q[3]}}}}}}}}}'
        ]

        self.get_logger().info(f'Sending goal to navigate to the specified pose: x={x}, y={y}, theta={theta} degrees')

        try:
            # Execute the command with a timeout of 20 seconds
            result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
            self.get_logger().info('Goal sent successfully.')
            self.get_logger().info(result.stdout)
        except subprocess.TimeoutExpired:
            self.get_logger().error('Navigation goal timed out after 120 seconds. Aborting the goal.')
            # Extract goal_id from the previous command if possible
            # Here, assume goal_id is not available and log the abort action
            abort_command = [
                'ros2', 'action', 'cancel', '--all', 'nav2_msgs/action/NavigateToPose'
            ]
            abort_result = subprocess.run(abort_command, capture_output=True, text=True)
            if abort_result.returncode == 0:
                self.get_logger().info('All navigation goals aborted successfully.')
                self.get_logger().info(abort_result.stdout)
            else:
                self.get_logger().error('Failed to abort navigation goals.')
                self.get_logger().error(abort_result.stderr)
        except subprocess.CalledProcessError as e:
            self.get_logger().error('Failed to send goal.')
            self.get_logger().error(e.stderr)

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)  
    number_of_bots = NUM_BOTS
    exploration_env = ExplorationEnv(number_of_bots)
    try:
        rclpy.spin(exploration_env)
    except KeyboardInterrupt:
        pass
    exploration_env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
