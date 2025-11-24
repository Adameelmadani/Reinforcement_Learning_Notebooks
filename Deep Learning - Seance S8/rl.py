import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame # Optional: for a visual render

class CustomGridWorldEnv(gym.Env):
    """
    Custom Grid World Environment
    
    - Agent: The learner
    - Goal: The target the agent wants to reach
    - Barriers: Obstacles the agent cannot pass through
    
    Observation Space: Dict('agent': [x, y], 'goal': [x, y])
    Action Space: Discrete(4) [0:Up, 1:Down, 2:Left, 3:Right]
    """
    
    # Optional: for pygame rendering
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, size=5, barriers=None, moving_goal=False, render_mode=None):
        super().__init__()
        
        self.size = size
        self.moving_goal = moving_goal
        
        # Define barriers. If None, create an empty list.
        # Barriers should be a list of (row, col) tuples.
        self.barriers = barriers if barriers is not None else []
        
        # Check for invalid barriers (outside grid)
        for b in self.barriers:
            if not (0 <= b[0] < size and 0 <= b[1] < size):
                raise ValueError(f"Barrier {b} is outside the grid bounds.")

        # --- Define Action and Observation Spaces ---
        
        # 4 actions: 0:Up, 1:Down, 2:Left, 3:Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: A dictionary with agent's and goal's location
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
            "goal": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
        })
        
        # --- Internal State ---
        self._agent_location = None
        self._goal_location = None
        
        # --- Rendering (Optional) ---
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.pix_square_size = 100 # Size of each grid cell in pixels

    def _get_obs(self):
        """Helper to get the current observation dictionary."""
        return {"agent": self._agent_location, "goal": self._goal_location}

    def _get_info(self):
        """Helper to get auxiliary info (distance to goal)."""
        return {"distance": np.linalg.norm(self._agent_location - self._goal_location, ord=1)}

    def _is_valid_location(self, loc):
        """Check if a location is valid (not a barrier)."""
        return list(loc) not in [list(b) for b in self.barriers]

    def _place_agent_and_goal(self):
        """Place agent and goal in valid, non-overlapping locations."""
        # Get all possible locations
        all_locs = [[r, c] for r in range(self.size) for c in range(self.size)]
        
        # Filter out barrier locations
        valid_locs = [loc for loc in all_locs if self._is_valid_location(loc)]
        
        if len(valid_locs) < 2:
            raise ValueError("Not enough valid locations to place agent and goal. Check barrier list.")

        # Sample two distinct locations
        chosen_indices = self.np_random.choice(len(valid_locs), size=2, replace=False)
        self._agent_location = np.array(valid_locs[chosen_indices[0]])
        self._goal_location = np.array(valid_locs[chosen_indices[1]])


    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed) # Important for reproducibility
        
        # Place agent and goal
        self._place_agent_and_goal()
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """Execute one time step in the environment."""
        
        # Map action to a direction vector [row, col]
        # 0:Up, 1:Down, 2:Left, 3:Right
        action_map = {
            0: np.array([-1, 0]), # Up
            1: np.array([1, 0]),  # Down
            2: np.array([0, -1]), # Left
            3: np.array([0, 1]),  # Right
        }
        direction = action_map[action]
        
        # Calculate potential new location
        new_location = self._agent_location + direction
        
        # --- Check for Collisions ---
        
        # 1. Check barrier collision
        if not self._is_valid_location(new_location):
            # Stay in the same place if hitting a barrier
            new_location = self._agent_location
            
        # 2. Check boundary collision (clip to stay within grid)
        new_location = np.clip(new_location, 0, self.size - 1)
        
        # Update agent location
        self._agent_location = new_location
        
        # --- Handle Moving Goal (if enabled) ---
        if self.moving_goal:
            # Randomly move goal to an adjacent cell (or stay put)
            move = self.np_random.choice(5) # 0:Up, 1:Down, 2:Left, 3:Right, 4:Stay
            if move < 4:
                goal_move_dir = action_map[move]
                new_goal_loc = self._goal_location + goal_move_dir
                
                # Check if new goal location is valid (not a barrier and in bounds)
                if (0 <= new_goal_loc[0] < self.size and
                    0 <= new_goal_loc[1] < self.size and
                    self._is_valid_location(new_goal_loc)):
                    self._goal_location = new_goal_loc

        # --- Calculate Reward and Termination ---
        terminated = np.array_equal(self._agent_location, self._goal_location)
        
        if terminated:
            reward = 10.0 # High positive reward for reaching the goal
        else:
            # Reward shaping: give small negative reward based on distance
            # This encourages the agent to move closer.
            reward = -np.linalg.norm(self._agent_location - self._goal_location, ord=1) / (self.size * 2)
            
        truncated = False # We don't have a time limit here, but could add one
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi":
            # Create a text-based grid
            grid = np.full((self.size, self.size), " . ")
            for b in self.barriers:
                grid[b[0], b[1]] = " B " # Barrier
            grid[self._goal_location[0], self._goal_location[1]] = " G " # Goal
            grid[self._agent_location[0], self._agent_location[1]] = " A " # Agent
            
            # Print row by row
            output = ""
            for row in grid:
                output += "".join(row) + "\n"
            print(output)
            
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        """Internal helper for pygame rendering."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.size * self.pix_square_size, self.size * self.pix_square_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(
            (self.size * self.pix_square_size, self.size * self.pix_square_size)
        )
        canvas.fill((255, 255, 255)) # White background

        # Draw barriers
        for b in self.barriers:
            pygame.draw.rect(
                canvas,
                (0, 0, 0), # Black
                pygame.Rect(
                    b[1] * self.pix_square_size,
                    b[0] * self.pix_square_size,
                    self.pix_square_size,
                    self.pix_square_size,
                ),
            )
            
        # Draw the goal
        pygame.draw.rect(
            canvas,
            (0, 255, 0), # Green
            pygame.Rect(
                self._goal_location[1] * self.pix_square_size,
                self._goal_location[0] * self.pix_square_size,
                self.pix_square_size,
                self.pix_square_size,
            ),
        )
        
        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255), # Blue
            (
                (self._agent_location[1] + 0.5) * self.pix_square_size,
                (self._agent_location[0] + 0.5) * self.pix_square_size,
            ),
            self.pix_square_size / 3,
        )

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (128, 128, 128), # Gray
                (0, x * self.pix_square_size),
                (self.size * self.pix_square_size, x * self.pix_square_size),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (128, 128, 128), # Gray
                (x * self.pix_square_size, 0),
                (x * self.pix_square_size, self.size * self.pix_square_size),
                width=1,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Close any open resources (like pygame window)."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
from gymnasium.envs.registration import register

register(
     id="CustomGridWorld-v0",
     entry_point="__main__:CustomGridWorldEnv", # Or: "my_project.envs:CustomGridWorldEnv"
     max_episode_steps=200, # Optional: set a max step limit
)

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
# Make sure your CustomGridWorldEnv class is defined or imported above
# and that the 'register' code has run.

# --- 1. Define your custom environment setup ---
grid_size = 5
my_barriers = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 2), (3, 3)]

# --- 2. Create the environment ---
# You can pass your custom args directly into gym.make()
env = gym.make(
    "CustomGridWorld-v0",
    size=grid_size,
    barriers=my_barriers,
    moving_goal=False, # Try False first, then True!
    render_mode="human" # Use 'human' to watch it train
)

# Optional: Check if the environment follows the Gymnasium API
# check_env(env) # This is a good sanity check

# --- 3. Instantiate the Agent ---
# DQN is a great choice for this. 'MultiInputPolicy' works with Dict spaces.
model = DQN(
    "MultiInputPolicy", # Changed from "MlpPolicy"
    env,
    verbose=1,
    tensorboard_log="./gridworld_tensorboard/"
)

# --- 4. Train the Agent ---
print("--- STARTING TRAINING ---")
model.learn(total_timesteps=20000, log_interval=4, progress_bar=True)
model.save("dqn_gridworld")
print("--- TRAINING COMPLETE ---")

# --- 5. Test the trained Agent ---
del model # remove to demonstrate loading
model = DQN.load("dqn_gridworld")

print("--- TESTING TRAINED AGENT ---")
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print("Goal reached!" if terminated else "Episode truncated.")
        obs, info = env.reset()

env.close()