import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from std_msgs.msg import String
import threading
import heapq

class TaskManager:
    def __init__(self):
        self.priority_queue = []  # Min-heap priority queue
        self.lock = threading.Lock()

    def add_task(self, urgency, time_of_arrival, pose):
        with self.lock:
            heapq.heappush(self.priority_queue, (-urgency, time_of_arrival, pose))

    def pop_task(self):
        with self.lock:
            return heapq.heappop(self.priority_queue) if self.priority_queue else None

    def is_empty(self):
        with self.lock:
            return len(self.priority_queue) == 0


class ClosestRobotSelector(Node):
    def __init__(self):
        super().__init__('closest_robot_selector')

        # Task management
        self.task_manager = TaskManager()
        self.time_of_arrival = 0

        # Action clients for computing paths
        self.compute_clients = {}
        self.distances = {}
        self.completed_paths = threading.Event()

        # Robot status management
        self.robot_status = {}
        self.robot_status_lock = threading.Lock()

        # ROS2 Subscriptions and Publishers
        self.create_subscription(String, 'robot_status', self.robot_status_callback, 10)
        self.create_subscription(PoseArray, 'goal_pose', self.goal_callback, 10)
        self.closest_robot_publisher = self.create_publisher(String, 'closest_robot', 10)

    def robot_status_callback(self, msg):
        """Update robot statuses."""
        with self.robot_status_lock:
            self.robot_status = {}
            for robot_status in msg.data.split(", "):
                if robot_status:
                    name, status = robot_status.split(": ")
                    self.robot_status[name] = status.lower()

    def goal_callback(self, msg):
        """Handle new goals."""
        for pose in msg.poses:
            urgency = 1 if self.is_urgent(pose) else 0  # Example condition for urgency
            self.task_manager.add_task(urgency, self.time_of_arrival, pose)
            self.time_of_arrival += 1

        threading.Thread(target=self.process_tasks, daemon=True).start()

    def process_tasks(self):
        """Process tasks and assign to robots."""
        while not self.task_manager.is_empty():
            task = self.task_manager.pop_task()
            if task:
                urgency, _, pose = task
                robot = self.find_best_robot(urgency, pose)
                if robot:
                    self.assign_task(robot, pose)

    def find_best_robot(self, urgency, pose):
        """Find the best robot for the task."""
        with self.robot_status_lock:
            if urgency == 1:  # Urgent tasks
                available_robots = [
                    name for name, status in self.robot_status.items() if status in ("free", "non-urgent")
                ]
            else:  # Non-urgent tasks
                available_robots = [name for name, status in self.robot_status.items() if status == "free"]

        if not available_robots:
            self.get_logger().info("No available robots for the task.")
            return None

        # Send path computation goals to robots
        self.distances = {}
        self.completed_paths.clear()

        for robot in available_robots:
            self.send_compute_goal(robot, pose)

        self.completed_paths.wait()  # Wait until distances for all robots are computed

        if not self.distances:
            self.get_logger().info("No valid paths for any robot.")
            return None

        return min(self.distances, key=self.distances.get)

    def send_compute_goal(self, robot, pose):
        """Send ComputePathToPose goal to a robot."""
        if robot not in self.compute_clients:
            self.compute_clients[robot] = ActionClient(
                self,
                ComputePathToPose,
                f'/{robot}/compute_path_to_pose'
            )

        client = self.compute_clients[robot]

        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'{robot} ComputePathToPose server not available.')
            self.distances[robot] = float('inf')
            self.check_and_set_completed_paths()
            return

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = self.create_pose_stamped(pose)

        self.get_logger().info(f"Sending ComputePathToPose goal to {robot}.")
        client.send_goal_async(goal_msg).add_done_callback(
            lambda future: self.handle_compute_response(robot, future)
        )

    def handle_compute_response(self, robot, future):
        """Handle the response from the compute path action."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"{robot} ComputePathToPose goal was rejected.")
            self.distances[robot] = float('inf')
            self.check_and_set_completed_paths()
            return

        self.get_logger().info(f"{robot} ComputePathToPose goal accepted.")
        goal_handle.get_result_async().add_done_callback(
            lambda fut: self.process_compute_result(robot, fut)
        )

    def process_compute_result(self, robot, future):
        """Process the result from the path computation."""
        try:
            result = future.result().result
            if result and result.path:
                distance = self.compute_distance(result.path)
                self.distances[robot] = distance
                self.get_logger().info(
                    f"{robot} estimated path distance: {distance:.2f}"
                )
            else:
                self.get_logger().error(f"Failed to compute path for {robot}.")
                self.distances[robot] = float('inf')
        except Exception as e:
            self.get_logger().error(f"Exception while processing result for {robot}: {e}")
            self.distances[robot] = float('inf')

        self.check_and_set_completed_paths()

    def compute_distance(self, path):
        """Compute the total distance of the path."""
        distance = 0.0
        poses = path.poses
        for i in range(len(poses) - 1):
            p1 = poses[i].pose.position
            p2 = poses[i + 1].pose.position
            distance += ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5
        return distance

    def check_and_set_completed_paths(self):
        """Check if all distances have been computed."""
        with self.robot_status_lock:
            if len(self.distances) == len(self.robot_status):
                self.completed_paths.set()

    def assign_task(self, robot, pose):
        """Assign a task to a robot."""
        with self.robot_status_lock:
            if self.robot_status[robot] != "free":
                # Push the previous task back into the queue
                self.task_manager.add_task(0, self.time_of_arrival, self.get_mock_robot_previous_task(robot))
                self.time_of_arrival += 1
            self.robot_status[robot] = "urgent" if self.is_urgent(pose) else "non-urgent"
        self.publish_robot_task(robot, pose)

    def publish_robot_task(self, robot, pose):
        """Publish assigned task."""
        msg = String()
        msg.data = f"{robot}: x={pose.position.x}, y={pose.position.y}"
        self.closest_robot_publisher.publish(msg)
        self.get_logger().info(f"Task assigned to {robot}: {msg.data}")

    def is_urgent(self, pose):
        """Determine if a task is urgent."""
        # Example condition for urgency: based on pose coordinates
        return pose.position.x > 5.0  # Replace with actual urgency condition

    def create_pose_stamped(self, pose):
        """Helper function to create a PoseStamped message."""
        from geometry_msgs.msg import PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'map'
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = ClosestRobotSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

