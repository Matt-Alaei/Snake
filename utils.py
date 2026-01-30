import os

from typing import Optional, Callable

import pygame, random
from pygame.locals import *

import numpy as np

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results



def normalizer(input, max, min):
    new_input = (input - min) / (max - min)
    return new_input


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func



class Button:
    '''
    Creates button object.

    :param button_x: The x coordinate of where the button should be created.
    :param button_y: The y coordinate of where the button should be created.
    :param color: The color of (label of) the button.
    :param button_width: The width of the button.
    :param button_height: The height of the button.
    :param box-color: The color of the button. If none, a transparent button object is created.
    :param width: The width of the outline of the button.
    '''
    def __init__(self, button_x, button_y, color, button_width = 150, button_height = 70, box_color = None, width = 2):
        self.button_x = button_x
        self.button_y = button_y
        self.color = color
        self.box_color = box_color
        self.button_width = button_width
        self.button_height = button_height
        self.width = width
        self.music = "sound effects/click.mp3"

    def draw(self, display, text, text_x, text_y):
        self.text = text
        self.text_x = text_x
        self.text_y = text_y

        if self.box_color == None:
            font = pygame.font.Font(pygame.font.match_font("Times New Roman"), 40)
            button_surface = pygame.Surface((self.button_x, self.button_y), pygame.SRCALPHA)
            text = font.render(text, True, self.color)
            text_rect = text.get_rect(center=(self.button_x // 2, self.button_y // 2))
            
            if self.width != 0:
                pygame.draw.rect(display, self.color, (self.button_x, self.button_y, self.button_width, self.button_height), self.width, border_radius=5)
            
            button_surface.blit(text, text_rect)
            display.blit(text, (self.text_x, self.text_y))

        else:
            pygame.draw.rect(display, self.box_color, (self.button_x, self.button_y, self.button_width, self.button_height), border_radius=5)
            pygame.draw.rect(display, self.color, (self.button_x, self.button_y, self.button_width, self.button_height), self.width, border_radius=5)
            font = pygame.font.Font(pygame.font.match_font("Times New Roman"), 40)
            text = font.render(text, True, self.color)
            display.blit(text, (self.text_x, self.text_y))

    def play_music(self):
        pygame.mixer.music.load(self.music)
        pygame.mixer.music.play()

    def create_rect(self):
        return pygame.Rect((self.button_x, self.button_y), (self.button_width, self.button_height))


class Consumable:
    '''
    Creates a consumbale object for the snake.

    :param music: If none, no sound effect is played when the consumable is consumed.
    '''
    def __init__(self, music = None):
        self.x = random.randrange(0, 580, 20)
        self.y = random.randrange(0, 580, 20)
        self.music = music

    def get_coordinates(self):
        return np.array([self.x, self.y])

    def new_coordinates(self):
        self.x = random.randrange(0, 580, 20)
        self.y = random.randrange(0, 580, 20)

    def eaten(self):
        if self.music != None:
            pygame.mixer.music.load(self.music)
            pygame.mixer.music.play()
        self.new_coordinates()


class Snake:
    '''
    Creates snake object.

    :param snake_x_speed: Snake's speed in the x axis.
    :param snake_y_speed: Snake's speed in the y axis.
    :param snake_x: Snake's head coordinate in the x axis.
    :param snake_y: Snake's head coordinate in the y axis.
    :param snake_width: Width of each body part of the snake.
    :param snake_height: Height of each body part of the snake.
    '''
    def __init__(self, snake_x_speed = 20, snake_y_speed = 0, snake_x = 380, snake_y = 280,
                 snake_width=20, snake_height=20):

        self.snake_x_speed = snake_x_speed
        self.snake_y_speed = snake_y_speed
        self.snake_x = snake_x
        self.snake_y = snake_y
        self.snake_list = []
        self.snake_length = 0
        self.height = snake_height
        self.width = snake_width

    def get_coords(self):
        return np.array([self.snake_x, self.snake_y])

    def move(self):
        self.snake_x += self.snake_x_speed
        self.snake_y += self.snake_y_speed

    def rotate(self, theta):
        velocity_vec = np.array([self.snake_x_speed, self.snake_y_speed])
        
        if theta == 90:
            rotation_vec = np.array([[0, -1],
                                    [1, 0]])
        else:
            rotation_vec = np.array([[0, 1],
                                [-1, 0]])
        
        self.snake_x_speed, self.snake_y_speed = np.dot(velocity_vec, rotation_vec)

    def snake_growing(self):
        '''
        Draws the snake and checks if the snake has collided with its body. If it has collided returns True, 
        otherwise returns False.
        '''

        g_over = False
        snake_head = [self.snake_x, self.snake_y]
        self.snake_list.append(snake_head)

        for member in self.snake_list[:-1]:
            if member == snake_head:
                g_over = True
        return g_over
    

    def render(self, display, snake_color, border_color=(0,0,0)):
        '''
        Renders the snake.
        '''
        for lst in self.snake_list:
            pygame.draw.rect(display, snake_color, (lst[0], lst[1], self.height, self.width), border_radius=5)
            pygame.draw.rect(display, border_color, (lst[0], lst[1], self.height, self.width), width=2, border_radius=5)


    def food_prox(self, food_x, food_y):
        '''
        Returns the location of the food relative to the snake's head.
        
        :param food_x: the x cooridnate of the food
        :param food_y: the y cooridnate of the food
        '''
        above, right, below, left, same_x, same_y = 0, 0, 0, 0, 0, 0
        
        if food_x > self.snake_x:
            right = 1
        elif food_x < self.snake_x:
            left = 1
        else:
            same_x = 1
        
        if food_y > self.snake_y:
            below = 1
        elif food_y < self.snake_y:
            above = 1
        else:
            same_y = 1

        return above, right, below, left, same_x, same_y
    
    def move_dir(self, food_x, food_y):
        '''
        Indicates whether snake is moving towards the food or not. Returns 1 if moving towards the food,
        0 otherwise.
        
        :param food_x: the x cooridnate of the food
        :param food_y: the y cooridnate of the food
        '''
        above, right, below, left, _, _ = self.food_prox(food_x, food_y)

        if above == 1 and self.snake_y_speed < 0:
            return 1
        elif right == 1 and self.snake_x_speed > 0:
            return 1
        elif below == 1 and self.snake_y_speed > 0:
            return True
        elif left == 1 and self.snake_x_speed < 0:
            return 1
        else:
            return 0
        
    def danger_dir(self):
        '''
        Returns the direction that if the snake moves in that direction
        snake will lose.
        '''
        danger_ahead, danger_right, danger_left = 0, 0, 0
        
        if len(self.snake_list) > 2:
            for body in self.snake_list:
                if self.snake_x_speed > 0:
                    if body[1] == self.snake_y and body[0] - self.snake_x == 20:
                        danger_ahead = 1
                    
                    elif body[0] == self.snake_x and body[1] - self.snake_y == 20:
                        danger_right = 1

                    elif body[0] == self.snake_x and self.snake_y - body[1] == 20:
                        danger_left = 1

                if self.snake_x_speed < 0:
                    if body[1] == self.snake_y and self.snake_x - body[0] == 20:
                        danger_ahead = 1
                    
                    elif body[0] == self.snake_x and body[1] - self.snake_y == 20:
                        danger_left = 1

                    elif body[0] == self.snake_x and self.snake_y - body[1] == 20:
                        danger_right = 1

                if self.snake_y_speed > 0:
                    if body[0] == self.snake_x and body[1] - self.snake_y == 20:
                        danger_ahead = 1
                    
                    elif body[1] == self.snake_y and body[0] - self.snake_x == 20:
                        danger_left = 1

                    elif body[1] == self.snake_y and self.snake_x - body[0] == 20:
                        danger_right= 1
                        
                if self.snake_y_speed < 0:
                    if body[0] == self.snake_x and self.snake_y - body[1] == 20:
                        danger_ahead = 1
                    
                    elif body[1] == self.snake_y and body[0] - self.snake_x == 20:
                        danger_right = 1

                    elif body[1] == self.snake_y and body[0] - self.snake_x == 20:
                        danger_left = 1

        return danger_ahead, danger_right, danger_left
    

class EnvSnake(gym.Env):
    '''
    Creates a gym Env object so the snake can interact with.

    :param height: The height of the environment.
    :param width: The width of the environment.
    :param time_limit: The max amount of time (steps) the snake can take to consume the apple.
    :param obs_shape: The shape of the observation space.
    '''
    def __init__(self, height=600, width=600, time_limit=150, obs_shape=8):

        self.height = height
        self.width = width
        
        self.limit = time_limit
        self.steps = 0

        self.snake = Snake()
        self.apple = Consumable()

        self.snake_coords = self.snake.get_coords()
        self.apple_coords = self.apple.get_coordinates()

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_shape))
        self.action_space = gym.spaces.Discrete(3)

    
    def _get_obs(self):
        '''
        Returns the state of the environemnt, which includes the normalized distance between the apple
        and the snake; the normalized distances of the snake from the boundaries of the environment; at which direction 
        danger lies (if movement along that direction is continued, the snake loses).
        '''
        
        state = []

        max_height = self.height-self.snake.height
        max_width = self.width-self.snake.width
        max_dist = np.sqrt(max_height**2 + max_width**2)
        
        dist_apple = np.linalg.norm(self.snake_coords-self.apple_coords)
        norm_dist_apple = normalizer(dist_apple, max_dist, 0)

        coord_walls = np.array([[0, 0], 
                                [self.height, self.width]])
        dist_walls = np.abs(coord_walls-self.snake_coords)
        norm_dist_walls = np.zeros(dist_walls.shape)
        norm_dist_walls[:,0] = normalizer(dist_walls[:,0], max_height, 0)
        norm_dist_walls[:,1] = normalizer(dist_walls[:,1], max_width, 0)

        danger_dir = list(self.snake.danger_dir())

        state.append(norm_dist_apple)
        state.extend(norm_dist_walls.flatten())
        state.extend(danger_dir)

        return np.array(state)
    

    def _calculate_reward(self):
        terminated = False
        truncated = False
        
        if self.snake.snake_list > self.snake.snake_length:
            self.snake.snake_list.pop(0)
        
        if np.array_equal(self.snake_coords, self.apple_coords):
            reward = 1

            self.apple.new_coordinates()
            self.apple_coords = self.apple.get_coordinates()

        elif self.steps > self.limit:
            reward = -2
            truncated = True

        elif self.snake.snake_growing():
            reward = -.5
            terminated = True

        elif self.snake_coords[0] < 0 or self.snake_coords[0] > (self.height-self.snake.height) \
            or self.snake_coords[1] < 0 or self.snake_coords[1] >= (self.width-self.snake.width):
            reward = -1
            terminated = True

        else:
            self.steps += 1
            reward = -.05

        return reward, terminated, truncated
    
    
    def step(self, action):
        if action == 0:
            self.snake.rotate(90)
            
        elif action == 1:
            self.snake.rotate(-90)

        self.snake.move()
        self.snake_coords = self.snake.get_coords()

        reward, terminated, truncated = self._calculate_reward()

        return self._get_obs(), reward, terminated, truncated, {}


    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed)

        self.snake = Snake()
        self.apple = Consumable()

        self.snake_coords = self.snake.get_coords()
        self.apple_coords = self.apple.get_coordinates()

        return self._get_obs(), {}


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


class PPOAgent:
    '''
    Creates a PPO agent.

    :params all params: all params are the same as the params used in the SB3 implementation of PPO.
    '''
    def __init__(self, env, policy, inital_lr, final_lr, n_steps, batch_size, n_epochs,
                 gamma, gae_lambda, clip_range, clip_range_vf, ent_coef, vf_coef, policy_kwargs,
                 seed, device='cpu', trained_path=None):

        self.train_log_dir = f'logs/train logs'
        os.makedirs(self.train_log_dir, exist_ok=True)

        # create directory to save validation logs
        self.val_log_dir = f'logs/val logs'
        os.makedirs(self.val_log_dir, exist_ok=True)

        # create directory the best model (of a time window and a seed)
        self.model_dir = f'models'
        os.makedirs(self.model_dir, exist_ok=True)

        self.env = env
        if trained_path is not None and os.path.exists(trained_path):
            self.ppo = PPO.load(trained_path, self.env, device)
        
        else:
            self.ppo = PPO(env, policy, linear_schedule(inital_lr, final_lr), n_steps, batch_size, n_epochs,
                  gamma, gae_lambda, clip_range, clip_range_vf, ent_coef, vf_coef,
                  policy_kwargs, seed, device, self.train_log_dir)

        
    def train(self, total_timesteps, eval_freq):
        callbacks = SaveOnBestTrainingRewardCallback(eval_freq=eval_freq, log_path=self.val_log_dir)

        self.ppo.learn(total_timesteps, callbacks, progress_bar=True)