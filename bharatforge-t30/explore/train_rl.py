import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

NUM_BOTS = 4

# Environment parameters
MAX_TIMESTEPS = 300
MAX_RANGE = 200

# Rewards
REWARD_NEW_PIXEL = 5.0
PENALTY_COLLISION = -100
PENALTY_OUT_OF_BOUNDS = -1000

class ExplorationEnv:
    def __init__(self, number_of_bots=5):
        self.number_of_bots = number_of_bots
        self.generate_map()
        self.explored_map = np.full_like(self.actual_map, -1)  # -1: Unexplored
        self.bot_positions = []
        for idx in range(self.number_of_bots):
            initial_position = self.get_random_position()
            self.bot_positions.append(initial_position)
            self.explored_map[initial_position] = 99 - idx
        self.steps = 0

    def generate_map(self):
        # Initialize the map as unknown (-1)
        self.actual_map = np.full((900, 900), -1, dtype=np.int8)

        # Generate a random polygon that occupies at least 50% of the map
        while True:
            num_vertices = np.random.randint(4, 10)
            angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
            radii = np.random.uniform(300, 450, num_vertices)
            x_center, y_center = 450, 450  # Center of the map
            x_vertices = (radii * np.cos(angles) + x_center).astype(np.int32)
            y_vertices = (radii * np.sin(angles) + y_center).astype(np.int32)
            polygon = np.array([x_vertices, y_vertices]).T

            # Create a mask for the polygon
            mask = np.zeros(self.actual_map.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            if np.sum(mask) >= 0.5 * self.actual_map.size:
                break

        # Add a 2-pixel thick boundary around the polygon
        boundary_mask = np.zeros_like(mask)
        cv2.polylines(boundary_mask, [polygon], isClosed=True, color=1, thickness=2)
        self.actual_map[boundary_mask.astype(bool)] = 100  # Set boundary as obstacle

        # Set the known area inside the polygon to 0 (free space)
        inner_mask = mask - boundary_mask
        self.actual_map[inner_mask.astype(bool)] = 0

        # Generate obstacles inside the polygon
        obstacles = []
        while True:
            obs_size = np.random.randint(20, 41)
            x = np.random.randint(0, self.actual_map.shape[0] - obs_size)
            y = np.random.randint(0, self.actual_map.shape[1] - obs_size)

            obs_rect = np.array([[x, y], [x + obs_size, y], [x + obs_size, y + obs_size], [x, y + obs_size]])
            # Check if obstacle is inside the inner polygon and does not overlap with existing obstacles
            obs_mask = np.zeros(self.actual_map.shape, dtype=np.uint8)
            cv2.fillPoly(obs_mask, [obs_rect], 1)
            obs_mask = obs_mask.astype(bool)

            if not np.all(inner_mask[obs_mask]):
                continue  # Obstacle is not entirely inside the polygon

            # Check for minimum distance of 50 pixels from other obstacles
            too_close = False
            for ox, oy, osize in obstacles:
                if (abs(x - ox) < 50 + obs_size) and (abs(y - oy) < 50 + obs_size):
                    too_close = True
                    break
            if too_close:
                continue

            # No overlap with other obstacles
            if np.any(self.actual_map[obs_mask] == 100):
                continue

            # Place the obstacle
            self.actual_map[obs_mask] = 100  # Obstacle
            obstacles.append((x, y, obs_size))

            # Stop if we've filled up to 50% of the inner area with obstacles
            if len(obstacles) * max(40, 40)**2 >= np.sum(inner_mask) * 0.05:
                break

        # Spawn bots inside the polygon on free pixels
        self.bot_positions = []
        for _ in range(self.number_of_bots):
            while True:
                x_spawn = np.random.randint(0, self.actual_map.shape[0])
                y_spawn = np.random.randint(0, self.actual_map.shape[1])
                if inner_mask[x_spawn, y_spawn] and self.actual_map[x_spawn, y_spawn] == 0:
                    self.bot_positions.append((x_spawn, y_spawn))
                    break

        # Initialize explored map
        self.explored_map = np.full_like(self.actual_map, -1, dtype=np.int8)
        self.max_area = np.sum(inner_mask)
        self.mask = mask

    def get_random_position(self):
        free_positions = np.argwhere(self.actual_map == 0)
        idx = np.random.randint(len(free_positions))
        return tuple(free_positions[idx])

    def get_state(self):
        return self.explored_map.copy()

    def update_map(self, idx):
        x, y = self.bot_positions[idx]
        self.explored_map[x, y] = 99 - idx  # Mark the bot's position
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

    def smooth_explored_map(self):
        smoothed_map = self.explored_map.copy()
        kernel = np.ones((5, 5), np.uint8)
        smoothed_map = cv2.morphologyEx(smoothed_map.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return smoothed_map

    def is_done(self):
    # Condition 1: A polygon of area at least half of max_area has been formed

        # Create a mask of the explored free space
        free_space_mask = (self.explored_map == 0)
        
        # Find connected components in the free space
        num_labels, labels_im = cv2.connectedComponents(free_space_mask.astype(np.uint8))
        # Note: labels start from 1 (0 is background)
        
        # Calculate the area of each connected component
        max_component_area = 0
        for label in range(1, num_labels):
            component_mask = (labels_im == label)
            component_area = np.sum(component_mask)
            if component_area > max_component_area:
                max_component_area = component_area
        
        # Check if the largest component area is at least half of max_area
        if max_component_area >= 0.3 * self.max_area:
            polygon_condition = True
        else:
            polygon_condition = False
        
        # Condition 2: Ratio between boundaries is greater than 100

        # Create masks for different regions
        free_space_mask = (self.explored_map == 0)
        obstacle_mask = (self.explored_map == 100)
        unexplored_mask = (self.explored_map == -1)
        
        # Define a 3x3 structuring element for dilation
        struct_element = np.ones((3, 3), dtype=np.uint8)
        
        # Dilate the free space mask to find its neighbors
        free_space_dilated = cv2.dilate(free_space_mask.astype(np.uint8), struct_element, iterations=1)
        
        # Boundary between free space and obstacles (0-100 boundary)
        boundary_free_obstacle = free_space_dilated & obstacle_mask
        length_free_obstacle_boundary = np.sum(boundary_free_obstacle)
        
        # Boundary between free space and unexplored areas (-1-0 boundary)
        boundary_free_unexplored = free_space_dilated & unexplored_mask
        length_free_unexplored_boundary = np.sum(boundary_free_unexplored)
        
        # Avoid division by zero
        if length_free_unexplored_boundary == 0:
            ratio = float('inf')
        else:
            ratio = length_free_obstacle_boundary / length_free_unexplored_boundary
        
        # Check if the ratio exceeds 5
        if ratio > 5:
            ratio_condition = True
        else:
            ratio_condition = False

        print(f"Step {self.steps}: Polygon Condition = {polygon_condition}, Ratio = {ratio}")

        # Check both conditions
        if (polygon_condition and ratio_condition) or self.steps >= MAX_TIMESTEPS:
            return True
        else:
            return False

    def step(self, target_positions):
        prev_bot_positions = self.bot_positions.copy()
        rewards = [0.0 for _ in range(self.number_of_bots)]

        prev_explored_free_spaces = np.sum(self.explored_map == 0)

        for idx, (bot_position, target_position) in enumerate(zip(self.bot_positions, target_positions)):
            x, y = target_position

            if not (0 <= x < self.actual_map.shape[0] and 0 <= y < self.actual_map.shape[1]):
                rewards[idx] += PENALTY_OUT_OF_BOUNDS
                continue

            if self.actual_map[x, y] == -1 or self.actual_map[x, y] == 100:
                rewards[idx] += PENALTY_COLLISION
                continue

            self.bot_positions[idx] = (x, y)
            self.update_map(idx)

            current_explored_free_spaces = np.sum(self.explored_map == 0)
            new_explored = current_explored_free_spaces - prev_explored_free_spaces

            if new_explored > 0:
                for idx in range(self.number_of_bots):
                    prev_pos = prev_bot_positions[idx]
                    curr_pos = self.bot_positions[idx]
                    distance = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                    if distance > 0:
                        rewards[idx] += (REWARD_NEW_PIXEL * new_explored) / (self.number_of_bots * distance)
                    else:
                        rewards[idx] += (REWARD_NEW_PIXEL * new_explored) / self.number_of_bots

        for idx, bot_pos in enumerate(self.bot_positions):
            x, y = bot_pos
            in_line_of_sight = False
            angles = np.linspace(0, 2 * np.pi, 360)
            for angle in angles:
                for r in range(1, MAX_RANGE + 1):
                    nx = int(x + r * np.cos(angle))
                    ny = int(y + r * np.sin(angle))
                    if 0 <= nx < self.explored_map.shape[0] and 0 <= ny < self.explored_map.shape[1]:
                        if self.explored_map[nx, ny] == -1:
                            dist = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
                            if dist <= 20:
                                rewards[idx] += 1000
                            in_line_of_sight = True
                            break
                        elif self.explored_map[nx, ny] == 100:
                            break
                    else:
                        break
                if in_line_of_sight:
                    break

        for i in range(self.number_of_bots):
            for j in range(i + 1, self.number_of_bots):
                dist = np.linalg.norm(np.array(self.bot_positions[i]) - np.array(self.bot_positions[j]))
                if dist <= 30:
                    rewards[i] += -10000
                    rewards[j] += -10000
                elif dist <= 100:
                    rewards[i] += -1000
                    rewards[j] += -1000

        self.steps += 1
        done = self.is_done()
        return self.get_state(), rewards, done, prev_bot_positions

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_bots):
        super(ActorCritic, self).__init__()
        self.num_bots = num_bots

        self.conv1 = nn.Conv2d(num_inputs, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((9, 9))

        self.fc = nn.Linear(64 * 9 * 9, 256)

        self.actor = nn.Linear(256, num_bots * 2)  # Output pred_x and pred_y for each bot
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

def main():
    num_bots = NUM_BOTS
    model = ActorCritic(num_inputs=4, num_bots=num_bots)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    plt.ion()
    for episode in range(100):
        env = ExplorationEnv(number_of_bots=num_bots)
        state = env.get_state()
        cumulative_reward = 0
        done = False
        step_count = 0
        while not done:
            step_count += 1
            state_channels = np.zeros((4, state.shape[0], state.shape[1]), dtype=np.float32)
            state_channels[0][state == -1] = 1.0  # Unexplored
            state_channels[1][state == 0] = 1.0   # Free space
            state_channels[2][state == 100] = 1.0  # Obstacles
            for idx in range(num_bots):
                state_channels[3][state == 99 - idx] = 1.0  # Bots

            state_tensor = torch.FloatTensor(state_channels)

            output, value = model(state_tensor)
            output = output.view(num_bots, 2)

            target_positions = []
            free_positions = np.argwhere(state == 0)  # Positions of free explored pixels

            if len(free_positions) > 0:
                map_height, map_width = state.shape
                normalized_free_positions = free_positions / np.array([map_height - 1, map_width - 1])

                for idx in range(num_bots):
                    pred_x = output[idx, 0].item() % 1.0
                    pred_y = output[idx, 1].item() % 1.0

                    # Compute distances to normalized free positions
                    distances = np.linalg.norm(normalized_free_positions - np.array([pred_x, pred_y]), axis=1)
                    nearest_idx = np.argmin(distances)
                    map_x, map_y = free_positions[nearest_idx]
                    target_positions.append((int(map_x), int(map_y)))
            else:
                # No free explored pixels available
                for idx in range(num_bots):
                    map_x, map_y = env.bot_positions[idx]
                    target_positions.append((int(map_x), int(map_y)))

            next_state, rewards, done, prev_bot_positions = env.step(target_positions)
            next_bot_positions = env.bot_positions

            cumulative_reward += sum(rewards)

            # Compute loss with tensor operations
            total_reward = torch.tensor(sum(rewards), dtype=torch.float32)
            advantage = total_reward - value.squeeze()

            policy_loss = -advantage
            value_loss = advantage.pow(2)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

             # Visualization
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.title("Actual Map")
            plt.imshow(env.actual_map, cmap='gray')
            for prev_pos, current_pos in zip(prev_bot_positions, next_bot_positions):
                plt.scatter(prev_pos[1], prev_pos[0], c='blue', marker='o')
                plt.scatter(current_pos[1], current_pos[0], c='red', marker='x')
            plt.legend(['Previous', 'Current'])

            plt.subplot(1, 2, 2)
            plt.title("Explored Map")

            # Map data values to indices for colormap
            display_map = np.copy(env.explored_map)
            display_map[display_map >= 99 - num_bots + 1] = 0  # Treat bot positions as free space
            value_to_index = {-1: 0, 0: 1, 100: 2}
            index_map = np.vectorize(value_to_index.get)(display_map)

            cmap = colors.ListedColormap(['gray', 'white', 'black'])
            plt.imshow(index_map, cmap=cmap, norm=colors.NoNorm())

            # Plot bot positions
            for prev_pos, current_pos in zip(prev_bot_positions, next_bot_positions):
                plt.scatter(prev_pos[1], prev_pos[0], c='blue', marker='o')
                plt.scatter(current_pos[1], current_pos[0], c='red', marker='x')
            plt.legend(['Previous', 'Current'])

            plt.pause(0.01)

            if step_count % 10 == 0:
                print(f"Episode {episode+1}, Step {step_count}, Cumulative Reward: {cumulative_reward}")

        print(f"Episode {episode+1}/100, Total Reward: {cumulative_reward}")
        #torch.save(model.state_dict(), f'weights/model{NUM_BOTS}.pth')

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
