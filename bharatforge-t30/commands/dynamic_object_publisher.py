#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time


class LogFileHandler(FileSystemEventHandler):
    def __init__(self, file_path, publisher):
        self.file_path = file_path
        self.publisher = publisher
        self.last_position = 0

        # Initialize the position at the end if the file already exists
        try:
            with open(self.file_path, 'r') as file:
                file.seek(0, 2)  # Move to the end of the file
                self.last_position = file.tell()
        except FileNotFoundError:
            # File doesn't exist initially, no need to set last_position
            pass

    def on_created(self, event):
        if event.src_path == self.file_path:
            self.last_position = 0  # Reset position for a newly created file

    def on_modified(self, event):
        if event.src_path == self.file_path:
            self.publish_new_entries()

    def publish_new_entries(self):
        """Read new lines added to the log file and publish them."""
        try:
            with open(self.file_path, 'r') as file:
                file.seek(self.last_position)  # Start reading from the last position
                new_lines = file.readlines()
                self.last_position = file.tell()  # Update the position for the next read

                for line in new_lines:
                    if line.strip():  # Skip empty lines
                        msg = String()
                        msg.data = line.strip()
                        self.publisher.publish(msg)
        except FileNotFoundError:
            # Handle case where file might be deleted temporarily
            self.last_position = 0


class DynamicObjectPublisher(Node):
    def __init__(self):
        super().__init__('dynamic_object_publisher')

        # Declare parameters and read them
        self.declare_parameter('log_file_path', 'detected_objects.txt')
        log_file_path = self.get_parameter('log_file_path').value

        # Create a ROS2 publisher
        self.publisher = self.create_publisher(String, 'detected_objects', 10)

        # Set up the watchdog observer
        self.event_handler = LogFileHandler(log_file_path, self.publisher)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, path=log_file_path, recursive=False)
        self.observer.start()

        self.get_logger().info(f"Started monitoring file: {log_file_path}")

    def continuously_publish(self):
        """Keep checking for new lines and publish them periodically."""
        while rclpy.ok():
            self.event_handler.publish_new_entries()  # Publish new lines from file
            time.sleep(0.5)  # Check for new lines every second

    def destroy_node(self):
        """Ensure the observer is stopped on node shutdown."""
        self.observer.stop()
        self.observer.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    dynamic_object_publisher = DynamicObjectPublisher()

    try:
        # Start continuous file checking and publishing in a separate thread
        dynamic_object_publisher.continuously_publish()
    except KeyboardInterrupt:
        dynamic_object_publisher.get_logger().info("Shutting down...")
    finally:
        dynamic_object_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

