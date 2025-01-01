#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        # Robot names or IDs (example: robot1, robot2, ..., robot8)
        self.robots = [f"robot{i}" for i in range(1, 9)]  # List of robots

        # Create dictionaries to hold the subscriptions and publishers for each robot
        self.robot_subscriptions = {}
        self.robot_publishers = {}

        # Initialize Kalman Filter parameters
        self.dt = 0.1  # Time step (adjust as needed)

        # State vector [x, y, vx, vy, ax, ay]
        self.state = np.zeros((6, 1))

        # State covariance matrix
        self.P = np.eye(6) * 1.0

        # State transition matrix
        self.A = np.array([
            [1, 0, self.dt, 0, 0.5 * self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5 * self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Control-input model (if any, here assumed none)
        self.B = np.zeros((6, 2))  # Example for acceleration control

        # Measurement matrix
        # Assuming we measure position, velocity, and acceleration directly
        self.H = np.eye(6)

        # Measurement noise covariance
        std_dev = 2e-4
        self.R = np.eye(6) * std_dev**2

        # Process noise covariance
        q = 1e-5
        self.Q = np.eye(6) * q

        self.last_time = self.get_clock().now()

        # Dynamically create subscriptions and publishers
        for robot in self.robots:
            self.robot_subscriptions[robot] = self.create_subscription(
                Odometry,
                f'/{robot}/odom',  # Dynamic subscription to robot-specific odom topic
                lambda msg, robot=robot: self.odom_callback(msg, robot),
                10
            )
            self.robot_publishers[robot] = self.create_publisher(
                Odometry,
                f'/{robot}/odom_filtered',  # Dynamic publisher to robot-specific filtered odom topic
                10
            )

    def odom_callback(self, msg, robot):
        # Time update
        current_time = self.get_clock().now()
        delta_time = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if delta_time <= 0:
            delta_time = self.dt  # Fallback to default

        # Update state transition matrix with new delta_time
        self.A = np.array([
            [1, 0, delta_time, 0, 0.5 * delta_time**2, 0],
            [0, 1, 0, delta_time, 0, 0.5 * delta_time**2],
            [0, 0, 1, 0, delta_time, 0],
            [0, 0, 0, 1, 0, delta_time],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Predict Step
        self.predict()

        # Measurement extraction from Odometry message
        # Position
        pos = msg.pose.pose.position
        x_meas = pos.x
        y_meas = pos.y

        # Orientation to extract angular velocity
        orientation_q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])

        # Linear velocity
        linear = msg.twist.twist.linear
        vx_meas = linear.x
        vy_meas = linear.y

        # Angular velocity
        angular = msg.twist.twist.angular
        wz_meas = angular.z

        # For simplicity, assume ax and ay are zero or can be derived
        # Here, we'll set acceleration measurements to zero
        ax_meas = 0.0
        ay_meas = 0.0

        # Construct measurement vector
        z = np.array([
            [x_meas],
            [y_meas],
            [vx_meas],
            [vy_meas],
            [ax_meas],
            [ay_meas]
        ])

        # Log raw measurements
        self.get_logger().info(
            f'Raw Measurements for {robot}: x={x_meas}, y={y_meas}, '
            f'vx={vx_meas}, vy={vy_meas}, ax={ax_meas}, ay={ay_meas}'
        )

        # Update Step
        self.update(z)

        # Log filtered state
        self.get_logger().info(f'Filtered State for {robot}: {self.state.flatten()}')

        # Publish the received message (exact data without filtering)
        self.publish_raw_odometry(robot, msg)

    def predict(self):
        # Predict the next state
        self.state = np.dot(self.A, self.state)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Compute Kalman Gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the state with measurement z
        y = z - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)

        # Update the covariance
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def publish_raw_odometry(self, robot, msg):
        # Publish the raw (unfiltered) odometry message
        self.robot_publishers[robot].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
