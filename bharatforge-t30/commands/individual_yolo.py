import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import atan2, degrees, cos, sin
import math
import argparse
import os

class ObjectCoordinatePublisher(Node):
    def __init__(self, robot_number):
        super().__init__('object_coordinate_publisher')

        # Dynamically construct topic names based on the robot number
        odom_topic = f'/robot{robot_number}/odom'
        rgb_topic = f'/robot{robot_number}/intel_realsense_r200_depth/image_raw'
        depth_topic = f'/robot{robot_number}/intel_realsense_r200_depth/depth/image_raw'
        camera_info_topic = f'/robot{robot_number}/intel_realsense_r200_depth/camera_info'

        # Odometry subscriber
        self.odom_subscriber = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

        # Subscribers for RGB image, depth image, and camera info
        self.rgb_subscriber = self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_image_callback,
            10
        )

        self.depth_subscriber = self.create_subscription(
            Image,
            depth_topic,
            self.depth_image_callback,
            10
        )

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )

        # Initialize variables
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.pixel_x = None
        self.pixel_y = None
        self.odom_position = None
        self.orientation=None

        # YOLO configuration
        self.config_path = 'yolov3.cfg'
        self.weights_path = 'yolov3.weights'
        self.classes_file_path = 'yolov3.txt'
        self.classes = None
        self.COLORS = None
        self.class_ids = []
        self.confidences = []
        self.boxes = []

        self.previous_position = None
        self.previous_detected = None
        self.log_file_path = 'detected_objects.txt'

        # Create the log file without writing anything
        open(self.log_file_path, 'w').close()
        self.get_logger().info(f"Log file created: {self.log_file_path}")

        # Load YOLO classes
        with open(self.classes_file_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def save_to_file(self, data):
        """Save data to a text file."""
        with open(self.log_file_path, 'a') as file:
            file.write(data + '\n')

    def odom_callback(self, msg):
        # Extract the robot's current position
        self.odom_position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = self.quaternion_to_euler(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )

        # Convert yaw to degrees
        yaw_degrees = degrees(yaw)
        self.orientation=yaw_degrees
        self.get_logger().info(f'Odom position: x={self.odom_position.x}, y={self.odom_position.y}, z={self.odom_position.z}, Orientation: {self.orientation}')

    def rgb_image_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_image_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.calculate_pixel()
        self.process_images()

    def calculate_pixel(self):
        if self.rgb_image is None:
            self.get_logger().warn("No RGB image received")
            return

        Width = self.rgb_image.shape[1]
        Height = self.rgb_image.shape[0]
        scale = 0.00392

        # Load YOLO network
        net = cv2.dnn.readNet(self.weights_path, self.config_path)

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(self.rgb_image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(self.get_output_layers(net))

        # Process detections
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    self.class_ids.append(class_id)
                    self.confidences.append(float(confidence))
                    self.boxes.append([x, y, w, h])

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, conf_threshold, nms_threshold)

        # Process valid detections
        for i in indices:
            i = i[0] if isinstance(i, np.ndarray) else i
            box = self.boxes[i]
            x, y, w, h = box
            cx, cy = int(x + w / 2), int(y + h / 2)
            self.pixel_x = cx
            self.pixel_y = cy
            self.draw_prediction(self.rgb_image, self.class_ids[i], self.confidences[i], round(x), round(y), round(x + w), round(y + h))

        # Display the processed RGB image
        cv2.imshow("YOLO Detection", self.rgb_image)
        cv2.waitKey(1)

    def quaternion_to_euler(self, x, y, z, w):
        """
        Convert a quaternion into Euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        # Compute Euler angles
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = atan2(t3, t4)

        return roll, pitch, yaw

    def process_images(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            self.get_logger().warn("Missing data for processing")
            return

        if self.pixel_x is None or self.pixel_y is None:
            self.get_logger().warn("No valid pixel coordinates")
            return

        m_fx = self.camera_info.k[0]
        m_fy = self.camera_info.k[4]
        m_cx = self.camera_info.k[2]
        m_cy = self.camera_info.k[5]
        inv_fx = 1.0 / m_fx
        inv_fy = 1.0 / m_fy

        if self.depth_image.dtype != np.float32:
            self.get_logger().warn("Depth image format mismatch (32FC1 expected)")
            return

        depth_value = self.depth_image[self.pixel_y, self.pixel_x]   # x aur y interchange kiye hai abhi ke liye

        if depth_value > 0:
            depth_in_meters = depth_value
            point_x = (self.pixel_x - m_cx) * depth_in_meters * inv_fx
            point_y = (self.pixel_y - m_cy) * depth_in_meters * inv_fy
            dist = math.sqrt(point_x ** 2 + point_y ** 2 + depth_in_meters ** 2)

            detected_object_name = "Unknown"
            if self.classes and self.class_ids:
                detected_object_name = self.classes[self.class_ids[-1]]

            # World coordinates
            if self.odom_position:
                object_z = self.odom_position.z 
                object_x = self.odom_position.x + dist * cos(self.orientation) + point_x * sin(self.orientation)  
                object_y = self.odom_position.y + dist * sin(self.orientation) - point_x * cos(self.orientation)

                # Log both world and robot-relative coordinates
                self.get_logger().info(
                    f'Detected {detected_object_name} at: '
                    f'World coordinates (x={object_x:.2f}, y={object_y:.2f}), '
                    f'Relative to robot (x={point_x:.2f}, y={point_y:.2f}), '
                    f'Distance: {dist:.2f}m'
                )

                # Save to file if condition is met
                if self.previous_position is not None:
                    prev_x, prev_y = self.previous_position
                    if detected_object_name != self.previous_detected and (abs(object_x - prev_x) >= 0.1 or abs(object_y - prev_y) >= 0.1):
                        data = (
                            f"Object: {detected_object_name}, "
                            f"World Coordinates: x={object_x:.2f}, y={object_y:.2f}, z={object_z:.2f}, "
                            f"Distance: {dist:.2f}m"
                        )
                        self.save_to_file(data)
                        self.get_logger().info('Saved in file')
                else:
                    data = (
                            f"Object: {detected_object_name}, "
                            f"World Coordinates: x={object_x:.2f}, y={object_y:.2f}, z={object_z:.2f}, "
                            f"Distance: {dist:.2f}m"
                        )
                    self.save_to_file(data)
                    self.get_logger().info('Saved in file')
                
                # Update previous position
                self.previous_position = (object_x, object_y)
                self.previous_detected = detected_object_name
        else:
            self.get_logger().info("Invalid depth value")

        

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f"{self.classes[class_id]}: {confidence:.2f}"
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        try:
            return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Object Coordinate Publisher')
    parser.add_argument('--robot', type=int, required=True, help='Robot number (e.g., 1 for /robot1)')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = ObjectCoordinatePublisher(robot_number=parsed_args.robot)
    rclpy.spin(node)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
