# llm.py

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import google.generativeai as genai
import json
import re

API_KEY = "AIzaSyAGAzglDEgQzr6X8FQP1Hu9CZkwhW-A_Zs"

class ObjectCoordinateProcessor(Node):
    def __init__(self, gui_queue):
        super().__init__('object_coordinate_processor')

        # Initialize the LLM model with an API key
        api_key = API_KEY  
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

        self.user_query = None  # Initialize user query

        # Initialize robot statuses as Idle (True)
        self.robot_positions = {}
        self.robot_positions_lock = threading.Lock()
        self.free_robots = {f'robot{i+1}': True for i in range(8)}
        self.free_robots_lock = threading.Lock()

        # Queue to communicate with the GUI
        self.gui_queue = gui_queue

        # Subscribe to robot positions
        for i in range(8):
            robot_name = f'robot{i+1}'
            self.create_subscription(
                Odometry,
                f'/{robot_name}/odom',
                lambda msg, robot=robot_name: self.update_robot_position(robot, msg),
                10
            )

        # Subscribe to object history
        self.create_subscription(String, 'object_history', self.listener_callback, 10)

        # Monitor when robots complete goals
        self.create_subscription(String, 'robot_reached_goal', self.robot_reached_goal_callback, 10)

        # Publisher for assigning robots
        self.closest_robot_publisher = self.create_publisher(String, 'closest_robot', 10)

        self.get_logger().info("ObjectCoordinateProcessor initialized.")

    def update_robot_position(self, robot, msg):
        """Update a robot's position."""
        with self.robot_positions_lock:
            self.robot_positions[robot] = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y
            )
        self.get_logger().info(f"Updated position of {robot}: {self.robot_positions[robot]}")

    def listener_callback(self, msg):
        """Process object data and user query."""
        try:
            object_data = json.loads(msg.data)
            self.get_logger().info(f"Received object data: {object_data}")
        except json.JSONDecodeError:
            self.get_logger().error("Failed to decode JSON from object_history")
            return

        # Send object data to GUI
        self.gui_queue.put(('update_goals', object_data.copy()))

        if not self.user_query:
            self.get_logger().warn("No user query provided.")
            return

        robot_positions = list(self.robot_positions.values())
        response_text = self.query_llm(self.user_query, robot_positions, object_data)

        if response_text:
            self.process_llm_response(response_text)
        else:
            self.get_logger().warn("No response from LLM.")

    def query_llm(self, query, robot_positions, object_data):
        """Query the LLM with object data and user query."""
        prompt = (
            "Here is a dictionary of objects and their coordinates:\n"
            f"{object_data}\n\n"
            "The current positions of robots are:\n"
            f"{robot_positions}\n\n"
            "Based on the user's query, identify the coordinates of the object in the form of (x, y) "
            "and the robot number in the form of robot number (z). "
            f"Query: {query}"
        )
        try:
            response = self.model.generate_content(prompt)
            self.get_logger().info(f"LLM response: {response.text}")
            return response.text
        except Exception as e:
            self.get_logger().error(f"Error querying LLM: {e}")
            return None

    def process_llm_response(self, response_text):
        """Process the LLM's response and assign a robot."""
        coordinate_pattern = r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        robot_number_pattern = r"robot\s*number\s*(\d+)"

        coord_match = re.search(coordinate_pattern, response_text)
        robot_match = re.search(robot_number_pattern, response_text)

        if coord_match and robot_match:
            x, y = map(float, coord_match.groups())
            robot_number = int(robot_match.group(1))

            robot_name = f'robot{robot_number}'
            with self.free_robots_lock:
                if self.free_robots.get(robot_name, False):
                    self.assign_robot_to_goal(robot_name, x, y)
                else:
                    self.get_logger().info(f"Robot '{robot_name}' is not free.")
        else:
            self.get_logger().info("Could not parse LLM response.")

    def assign_robot_to_goal(self, robot, x, y):
        """Assign a robot to a goal and update status."""
        # Mark robot as Busy
        with self.free_robots_lock:
            self.free_robots[robot] = False
        self.get_logger().info(f"Assigned {robot} to goal at ({x}, {y})")

        # Publish the assignment
        goal_msg = String()
        goal_msg.data = f"{robot}, Goal: x={x}, y={y}"
        self.closest_robot_publisher.publish(goal_msg)
        self.get_logger().info(f"Published: {goal_msg.data}")

        # Update GUI
        self.gui_queue.put(('update_robot_status', self.free_robots.copy()))
        self.gui_queue.put(('robot_assigned', {'robot': robot, 'goal': (x, y)}))

    def robot_reached_goal_callback(self, msg):
        """Update robot status when goal is reached."""
        robot_name = msg.data.split(' ')[0]
        with self.free_robots_lock:
            self.free_robots[robot_name] = True
        self.get_logger().info(f"{robot_name} marked as Idle after reaching the goal.")

        # Update GUI
        self.gui_queue.put(('update_robot_status', self.free_robots.copy()))

class EnhancedGUI:
    def __init__(self, master, processor, gui_queue):
        self.master = master
        self.processor = processor
        self.gui_queue = gui_queue
        self.master.title("Robot Coordinator")

        # Set window size and make it resizable
        self.master.geometry("800x700")
        self.master.resizable(True, True)

        # Create a ttk style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12), padding=6)
        self.style.configure("TLabel", font=("Arial", 10), anchor="w")
        self.style.configure("TLabelFrame", font=("Arial", 12), padding=(10, 5))
        self.style.configure("Treeview", font=("Arial", 10), rowheight=30)
        self.style.configure("Treeview.Heading", font=("Arial", 12))

        # Frame for User Query
        self.query_frame = ttk.LabelFrame(self.master, text="User Query", padding=(10, 5))
        self.query_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.query_label = ttk.Label(self.query_frame, text="Enter your query:")
        self.query_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.query_entry = ttk.Entry(self.query_frame, width=40, font=("Arial", 12))
        self.query_entry.grid(row=0, column=1, padx=5, pady=5)

        self.submit_button = ttk.Button(self.query_frame, text="Submit Query", command=self.submit_query)
        self.submit_button.grid(row=0, column=2, padx=5, pady=5)

        # Frame for Robot Status
        self.robot_frame = ttk.LabelFrame(self.master, text="Robot Status", padding=(10, 5))
        self.robot_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        self.robot_status_tree = ttk.Treeview(self.robot_frame, columns=("Robot", "Status"), show="headings")
        self.robot_status_tree.heading("Robot", text="Robot")
        self.robot_status_tree.heading("Status", text="Status")
        self.robot_status_tree.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Dictionary to map robot names to Treeview item IDs
        self.robot_items = {}

        # Initialize robot statuses
        self.initialize_robot_status()

        # Frame for Predefined Goal Status
        self.goal_frame = ttk.LabelFrame(self.master, text="Predefined Goal Status", padding=(10, 5))
        self.goal_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

        self.goal_status_tree = ttk.Treeview(self.goal_frame, columns=("Object", "Coordinates"), show="headings")
        self.goal_status_tree.heading("Object", text="Object")
        self.goal_status_tree.heading("Coordinates", text="Coordinates")
        self.goal_status_tree.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Frame for LLM Responses
        self.llm_frame = ttk.LabelFrame(self.master, text="LLM Response", padding=(10, 5))
        self.llm_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)

        self.llm_response_text = tk.Text(self.llm_frame, height=6, state='disabled')
        self.llm_response_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Configure grid weights for resizing
        self.master.grid_rowconfigure(2, weight=1)
        self.master.grid_rowconfigure(3, weight=1)
        self.master.grid_rowconfigure(4, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.robot_frame.grid_rowconfigure(0, weight=1)
        self.robot_frame.grid_columnconfigure(0, weight=1)

        self.goal_frame.grid_rowconfigure(0, weight=1)
        self.goal_frame.grid_columnconfigure(0, weight=1)

        self.llm_frame.grid_rowconfigure(0, weight=1)
        self.llm_frame.grid_columnconfigure(0, weight=1)

        # Start GUI update loop
        self.update_gui()

    def initialize_robot_status(self):
        """Initialize robot statuses as Idle and map robot names to Treeview items."""
        robots = [f'robot{i+1}' for i in range(8)]
        for i,robot in enumerate(robots):
            if i!=6:
                item_id = self.robot_status_tree.insert('', 'end', values=(robot, 'Idle'))
            else:
                item_id = self.robot_status_tree.insert('', 'end', values=(robot, 'Busy'))
            self.robot_items[robot] = item_id

    def submit_query(self):
        """Submit user query and set processor query."""
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Input Required", "Please enter a query.")
            return

        if self.processor:
            self.processor.user_query = query
            self.processor.get_logger().info(f"User query submitted: {query}")
            messagebox.showinfo("Query Submitted", f"Your query has been submitted: {query}")
        else:
            messagebox.showerror("Error", "Processor not initialized yet.")
        self.query_entry.delete(0, tk.END)
        self.query_entry.focus()

    def assign_robot3(self):
        """Assign Robot3 to a specified goal object."""
        object_name = self.assign_robot_entry.get().strip()
        if not object_name:
            messagebox.showwarning("Input Required", "Please enter a goal object name.")
            return

        # Retrieve the predefined goals from the GUI's goal_status_tree
        goals = {}
        for item in self.goal_status_tree.get_children():
            obj, coords = self.goal_status_tree.item(item, 'values')
            goals[obj] = coords

        # Check if the entered object exists
        if object_name not in goals:
            messagebox.showerror("Error", f"Object '{object_name}' not found in predefined goals.")
            return

        # Extract coordinates
        coord_text = goals[object_name]
        try:
            x, y = map(float, re.findall(r'-?\d+\.?\d*', coord_text))
        except ValueError:
            messagebox.showerror("Error", f"Invalid coordinates format for object '{object_name}'.")
            return

        # Assign Robot3
        robot_name = 'robot3'
        with self.processor.free_robots_lock:
            if not self.processor.free_robots.get(robot_name, False):
                messagebox.showinfo("Info", f"{robot_name} is already Busy.")
                return

        # Assign the robot
        self.processor.assign_robot_to_goal(robot_name, x, y)
        self.gui_queue.put(('robot_assigned', {'robot': robot_name, 'goal': (x, y)}))

        # Clear the entry
        self.assign_robot_entry.delete(0, tk.END)
        self.assign_robot_entry.focus()

    def update_gui(self):
        """Update the GUI with data from the queue."""
        try:
            while True:
                item = self.gui_queue.get_nowait()
                if item[0] == 'update_robot_status':
                    self.refresh_robot_status(item[1])
                elif item[0] == 'update_goals':
                    self.refresh_goals(item[1])
                elif item[0] == 'robot_assigned':
                    self.display_llm_response(item[1])
        except queue.Empty:
            pass
        # Schedule the next GUI update
        self.master.after(100, self.update_gui)  # Pass the function reference without parentheses

    def refresh_robot_status(self, robot_statuses):
        """Refresh the robot status tree view by updating each robot's status."""
        for robot, status in robot_statuses.items():
            status_text = 'Idle' if status else 'Busy'
            item_id = self.robot_items.get(robot)
            if item_id:
                self.robot_status_tree.set(item_id, "Status", status_text)

    def refresh_goals(self, goals_dict):
        """Refresh the predefined goal status tree view."""
        # Clear existing entries
        for item in self.goal_status_tree.get_children():
            self.goal_status_tree.delete(item)
        # Insert goals from the dictionary
        for obj_name, coords in goals_dict.items():
            coord_text = f'({coords["x"]}, {coords["y"]})'
            self.goal_status_tree.insert('', 'end', values=(obj_name, coord_text))

    def display_llm_response(self, data):
        """Display LLM response in the GUI."""
        robot = data['robot']
        goal = data['goal']
        response_text = f"Assigned {robot} to goal at position {goal}"
        self.llm_response_text.configure(state='normal')
        self.llm_response_text.insert('end', response_text + '\n')
        self.llm_response_text.configure(state='disabled')
        self.llm_response_text.see('end')

def main():
    rclpy.init()
    gui_queue = queue.Queue()

    # Start the ROS node in a separate thread
    def ros_thread():
        processor = ObjectCoordinateProcessor(gui_queue)
        gui.processor = processor  # Assign the processor to the GUI
        rclpy.spin(processor)
        processor.destroy_node()
        rclpy.shutdown()

    threading.Thread(target=ros_thread, daemon=True).start()

    # Start the Tkinter GUI
    root = tk.Tk()
    gui = EnhancedGUI(root, None, gui_queue)
    root.mainloop()

if __name__ == '__main__':
    main()