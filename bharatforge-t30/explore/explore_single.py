import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.ndimage import binary_erosion, binary_dilation
from collections import deque

# Environment parameters
MAX_AREA = 10000  # Maximum number of free cells
MAX_TIMESTEPS = 5000
MAX_RANGE = 10

# Rewards
REWARD_NEW_PIXEL = 1.0  # Adjusted for percentage
PENALTY_UNEXPLORED = -0.5
PENALTY_COLLISION = -10
PENALTY_OUT_OF_BOUNDS = -100

# Define the environment
class ExplorationEnv:
    def __init__(self):
        self.generate_map()
        self.explored_map = np.full_like(self.actual_map, -1)  # -1: Unexplored, 0: Free, 100: Obstacle, 5: Bot
        self.bot_position = self.get_random_position()
        self.explored_map[self.bot_position] = 5
        self.steps = 0

    def generate_map(self):
        grid_size = 100  # Define a large enough grid
        self.actual_map = np.ones((grid_size, grid_size), dtype=np.int32) * -1  # Initialize as invalid cells

        # Generate random polygon (can be convex or concave)
        num_vertices = np.random.randint(4, 15)
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        radius = np.random.uniform(20, 40, size=num_vertices)
        center = (grid_size // 2, grid_size // 2)
        points = np.vstack((center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles))).T

        # Create a path and fill the polygon
        xv, yv = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        xv = xv.flatten()
        yv = yv.flatten()
        coords = np.vstack((xv, yv)).T
        path = Path(points)
        mask = path.contains_points(coords)
        mask = mask.reshape((grid_size, grid_size))
        self.actual_map[mask] = 0  # Free space

        # Create a 2-pixel wide boundary around the polygon
        boundary = mask ^ binary_erosion(mask, structure=np.ones((3, 3)), iterations=1)
        boundary_2pixel = binary_dilation(boundary, structure=np.ones((3, 3)), iterations=1)
        self.actual_map[boundary_2pixel] = 100  # Set boundary as obstacles

        # Calculate max_area based on free space excluding the boundary obstacles
        self.max_area = np.sum(self.actual_map == 0)
        num_obstacles = int(0.2 * self.max_area)  # 20% obstacles
        free_positions = np.argwhere(self.actual_map == 0)
        if len(free_positions) < num_obstacles:
            num_obstacles = len(free_positions) // 2  # Prevent over-assigning
        obstacle_indices = free_positions[np.random.choice(len(free_positions), num_obstacles, replace=False)]
        for pos in obstacle_indices:
            self.actual_map[tuple(pos)] = 100  # Obstacle

    def get_random_position(self):
        free_positions = np.argwhere(self.actual_map == 0)
        idx = np.random.randint(len(free_positions))
        return tuple(free_positions[idx])

    def get_state(self):
        return self.explored_map.copy()

    def step(self, target_position):
        prev_bot_position = self.bot_position
        x, y = target_position

        # Calculate Euclidean distance between current and target positions
        distance = np.linalg.norm(np.array([x, y]) - np.array(self.bot_position))
        distance = max(distance, 1.0)  # Prevent division by zero

        # Check for out of bounds
        if not (0 <= x < self.actual_map.shape[0] and 0 <= y < self.actual_map.shape[1]):
            reward = PENALTY_OUT_OF_BOUNDS
            done = False
            return self.get_state(), reward, done, prev_bot_position

        # Check if target position is invalid or known obstacle
        if self.actual_map[x, y] == -1 or self.explored_map[x, y] == 100:
            reward = PENALTY_COLLISION
            done = False
            return self.get_state(), reward, done, prev_bot_position

        # Store explored area before moving
        prev_explored = np.sum((self.explored_map == 0) | (self.explored_map == 100))

        # Move to the target position
        self.bot_position = (x, y)
        self.update_map()

        # Calculate newly explored pixels
        new_explored = np.sum((self.explored_map == 0) | (self.explored_map == 100)) - prev_explored

        # If moved into an obstacle (discovered now)
        if self.actual_map[x, y] == 100:
            self.bot_position = prev_bot_position
            self.explored_map[x, y] = 100
            reward = PENALTY_COLLISION
        else:
            # Reward is new explored pixels divided by distance
            reward = REWARD_NEW_PIXEL * (new_explored / distance)

        done = self.is_done()
        self.steps += 1
        return self.get_state(), reward, done, prev_bot_position

    def update_map(self):
        x, y = self.bot_position
        self.explored_map[x, y] = 5  # Bot's position
        angles = np.linspace(0, 2 * np.pi, 360)
        for angle in angles:
            for r in range(1, MAX_RANGE + 1):
                nx = int(x + r * np.cos(angle))
                ny = int(y + r * np.sin(angle))
                if 0 <= nx < self.actual_map.shape[0] and 0 <= ny < self.actual_map.shape[1]:
                    if self.explored_map[nx, ny] == -1 and self.actual_map[nx, ny] != -1:
                        self.explored_map[nx, ny] = self.actual_map[nx, ny]
                    if self.actual_map[nx, ny] == 100:
                        break
                else:
                    break

    def calculate_reward(self):
        # This function is no longer used since reward is calculated in the step function
        pass

    def is_done(self):
        explored_area = np.sum((self.explored_map == 0) | (self.explored_map == 100))
        percentage_explored = explored_area / self.max_area

        # Perform flood fill from the robot's position to find reachable free cells
        visited = np.zeros_like(self.actual_map, dtype=bool)
        queue = deque([self.bot_position])
        visited[self.bot_position] = True

        while queue:
            current = queue.popleft()
            x, y = current
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.actual_map.shape[0] and
                    0 <= ny < self.actual_map.shape[1] and
                    not visited[nx, ny] and
                    self.actual_map[nx, ny] != 100):
                    visited[nx, ny] = True
                    queue.append((nx, ny))

        # Identify all reachable free cells
        reachable_free = (self.actual_map == 0) & visited
        # Check if all reachable free cells have been explored
        all_reachable_explored = np.all((self.explored_map == 0) | 
                                       (self.explored_map == 100) | 
                                       (self.explored_map == 5) | 
                                       (~reachable_free))

        if (percentage_explored >= 0.5) and all_reachable_explored or (self.steps >= MAX_TIMESTEPS):
            return True
        return False

# Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, input_size):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_size, 256)
        self.policy_head = nn.Linear(256, input_size)
        self.value_head = nn.Linear(256, 1)
        self.env = None
        self.current_position = None

    def forward(self, x):
        x = torch.relu(self.fc(x))
        logits = self.policy_head(x)

        # Mask invalid actions (unexplored, obstacles, or out of bounds)
        explored_free = (self.env.explored_map.flatten() == 0).astype(np.float32)
        mask = torch.tensor(explored_free, dtype=torch.float32)
        large_negative = -1e9  # Use a large negative value instead of -inf

        # Apply mask: invalid actions get a large negative value
        mask = torch.where(mask == 1, torch.tensor(0.0), torch.tensor(large_negative))

        policy_logits = logits + mask

        # Handle case where all actions are invalid
        if torch.all(mask == large_negative):
            # Assign uniform probabilities to allow at least some exploration
            policy = torch.softmax(logits, dim=-1)
        else:
            policy = torch.softmax(policy_logits, dim=-1)

        value = self.value_head(x)
        return policy, value

# Training parameters
NUM_EPISODES = 100
GAMMA = 0.99
LR = 1e-5

# Possible actions (all free pixels, action space is large)
# To manage large action spaces, consider using techniques like action masking or hierarchical actions

def main():
    model = ActorCritic(input_size=10000)  # Assuming max grid size is 100x100
    optimizer = optim.Adam(model.parameters(), lr=LR)

    plt.ion()
    for episode in range(NUM_EPISODES):
        env = ExplorationEnv()
        state = env.get_state()
        state_flat = state.flatten()
        log_probs = []
        values = []
        rewards = []
        cumulative_reward = 0
        done = False
        step_count = 0

        while not done:
            step_count += 1
            state_tensor = torch.FloatTensor(state_flat)
            model.env = env
            model.current_position = env.bot_position
            policy, value = model(state_tensor)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Convert action index to coordinates
            x, y = divmod(action.item(), env.actual_map.shape[1])
            current_bot_position = env.bot_position
            next_state, reward, done, _ = env.step((x, y))
            next_bot_position = env.bot_position

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            cumulative_reward += reward
            state_flat = next_state.flatten()

            if step_count % 100 == 0:
                print(f"Episode {episode+1}, Step {step_count}, Cumulative Reward: {cumulative_reward}")

            # Visualization
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.title("Actual Map")
            plt.imshow(env.actual_map, cmap='gray')
            plt.scatter(current_bot_position[1], current_bot_position[0], c='red', marker='o', label='Current')
            plt.scatter(next_bot_position[1], next_bot_position[0], c='yellow', marker='x', label='Next')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.title("Explored Map")
            cmap = colors.ListedColormap(['green', 'white', 'black'])
            bounds = [-1.5, -0.5, 0.5, 100.5]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            display_map = np.copy(env.explored_map)
            display_map[display_map == 5] = 0  # Treat bot position as free space
            plt.imshow(display_map, cmap=cmap, norm=norm)
            plt.scatter(current_bot_position[1], current_bot_position[0], c='red', marker='o')
            plt.scatter(next_bot_position[1], next_bot_position[0], c='yellow', marker='x')
            plt.legend(['Current', 'Next'])

            plt.pause(0.01)

        # Compute returns and losses
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)  # Ensure float32
        loss = 0
        for log_prob, value, Gt in zip(log_probs, values, returns):
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_loss = nn.functional.mse_loss(value, torch.tensor([Gt], dtype=torch.float32).to(value.device))
            loss += policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode+1}/{NUM_EPISODES}, Total Reward: {sum(rewards)}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()