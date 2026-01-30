import neat

import pygame, sys
from pygame.locals import *

import numpy as np

from stable_baselines3.common.env_util import make_vec_env

import torch

import pickle

from utils import Button, Snake, Consumable, normalizer, EnvSnake, PPOAgent


pygame.init()

light_red = (255, 183, 183)
blue_sky = (134, 225, 253)
red = (255, 0, 0)
green = (0, 255, 0)
New = (238, 232, 170)
yellow = (255, 255, 0)
white = (255,255,255)
black = (0,0,0)
orangish = (250, 167, 22)
dark_blue = (5, 30, 155)

snake_image = pygame.image.load("icons/another_snake.png")
background = pygame.image.load("icons/snake.png")
apple_image = pygame.image.load("icons/apple.png")
ice_image = pygame.image.load("icons/ice.png")
chili_image = pygame.image.load("icons/chili-pepper.png")

window_width = 600
window_height = 600
dis = pygame.display.set_mode((window_height, window_width))
pygame.display.set_caption("Snake")
pygame.display.set_icon(background)
clock = pygame.time.Clock()

    

class Game:

    highest_score = 0
    neat_highest_score = 0
    gen = 0
    snake_color = orangish
    background_color = New
    high_scores = []
    left_turns = []
    right_turns = []
    best_genome = None
    

    @staticmethod
    def main_menu():
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                dis.fill(Game.background_color)
                dis.blit(snake_image, (250, 470))

                font = pygame.font.Font(pygame.font.match_font("Times New Roman"), 100)
                text = font.render("Snake", True, Game.snake_color)
                dis.blit(text, (180, 20))

                start = Button(225, 150, Game.snake_color)
                start.draw(dis, "Start", 260, 160)

                options = Button(225, 265, Game.snake_color)
                options.draw(dis, "Options", 233, 275)
                
                AI = Button(225, 380, Game.snake_color)
                AI.draw(dis, "AI", 275, 390)

                pygame.display.flip()

                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                if start.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        start.play_music()
                        Game.game_opening()
            
                if options.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        options.play_music()
                        Game.options_menu()

                if AI.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        AI.play_music()
                        Game.AI_menu()
    
    
    
    @staticmethod
    def options_menu():
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                dis.fill(Game.background_color)

                font2 = pygame.font.Font(pygame.font.match_font("Times New Roman"), 100)
                text2 = font2.render("Themes", True, Game.snake_color)
                dis.blit(text2, (130, 110))
                back = Button(0, 0, Game.snake_color, width=0)
                back.draw(dis, "Back", 0, 0)
                red = Button(50, 265, (255, 0, 0), box_color=light_red)
                red.draw(dis, "Red", 90, 275)
                blue = Button(225, 265, blue_sky, box_color=(0, 0, 255))
                blue.draw(dis, "Blue", 264, 275)
                orange = Button(400, 265, (250, 167, 22), box_color=New)
                orange.draw(dis, "Orange", 418, 275)

                pygame.display.flip()

                mouse_x, mouse_y = pygame.mouse.get_pos()

                if red.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        red.play_music()
                        Game.snake_color = red.color
                        Game.background_color = red.box_color
                        Game.main_menu()
                if blue.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        blue.play_music()
                        Game.snake_color = blue.color
                        Game.background_color = blue.box_color
                        Game.main_menu()
                if orange.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        orange.play_music()
                        Game.snake_color = orange.color
                        Game.background_color = orange.box_color
                        Game.main_menu()

                if back.create_rect().collidepoint(mouse_x,mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        back.play_music()
                        Game.main_menu()
                if event.type == KEYDOWN:
                    if event.key == K_BACKSPACE:
                        back.play_music()
                        Game.main_menu()


    @staticmethod
    def AI_menu():
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                dis.fill(Game.background_color)

                back = Button(0, 0, Game.snake_color, width=0)
                back.draw(dis, "Back", 0, 0)

                neat_button = Button(225, 227.5, Game.snake_color)
                neat_button.draw(dis, "NEAT", 250, 237.5)

                ppo_button = Button(225, 302.5, Game.snake_color)
                ppo_button.draw(dis, 'PPO', 265, 312.5)

                pygame.display.flip()

                mouse_x, mouse_y = pygame.mouse.get_pos()

                if back.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        back.play_music()
                        Game.main_menu()
                if event.type == KEYDOWN:
                    if event.key == K_BACKSPACE:
                        back.play_music()
                        Game.main_menu()
                if neat_button.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        neat_button.play_music()
                        Game.AI_run('neat config')
                if ppo_button.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        ppo_button.play_music()
                        Game.PPO_menu()


    @staticmethod
    def PPO_menu():
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                dis.fill(Game.background_color)

                back = Button(0, 0, Game.snake_color, width=0)
                back.draw(dis, "Back", 0, 0)                                                     

                train_button = Button(225, 227.5, Game.snake_color)
                train_button.draw(dis, 'TRAIN', 240, 237.5)

                test_button = Button(225, 302.5, Game.snake_color)
                test_button.draw(dis, 'TEST', 255, 312.5)

                pygame.display.flip()

                mouse_x, mouse_y = pygame.mouse.get_pos()

                if back.create_rect().collidepoint(mouse_x, mouse_y):
                    if event.type == MOUSEBUTTONDOWN:
                        back.play_music()
                        Game.AI_menu()
                if event.type == KEYDOWN:
                    if event.key == K_BACKSPACE:
                        back.play_music()
                        Game.AI_menu()



    @staticmethod
    def train_ppo():
        # env hyper params
        pass
        # n_envs = 5
        # time_limit = 300

        # env_kwargs = {'time_limit': time_limit}

        # # PPO hyper params
        # policy = 'MlpPolicy'
        
        # inital_lr = 3e-4
        # final_lr = 1e-5

        # # n_steps = 
        # # batch_size = 
        # # n_epochs = 
        # # gamma = .999
        # # gae_lambda = 
        # # clip_range = 
        # # clip_range_vf = 
        # # ent_coef = 
        # # vf_coef = 
        
        # # net_arch = [32, 32]
        # # activation_fn = torch.nn.ReLU
        # # policy_kwargs = {'net_arch': net_arch, 'activation_fn': activation_fn}

        # # total_timesteps = 
        # # eval_freq = 

        # train_env = make_vec_env(EnvSnake, n_envs=n_envs, env_kwargs=env_kwargs)

        # ppo_agent = PPOAgent(train_env, policy=policy, inital_lr=inital_lr, final_lr=final_lr,
        #                      n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma,
        #                      gae_lambda=gae_lambda , clip_range=clip_range, clip_range_vf=clip_range_vf,
        #                      ent_coef=ent_coef, vf_coef=vf_coef, policy_kwargs=policy_kwargs)
        
        # ppo_agent.train(total_timesteps=total_timesteps, eval_freq=eval_freq)

        

    @staticmethod                        
    def AI_eval(genomes, config):
        Game.gen += 1
        num_left = 0
        num_right = 0

        for genome_id, genome in genomes:
            fps = 100
            
            genome.fitness = 0
            
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            snake = Snake()
            apple = Consumable()
            timer = 0
            left_turn = 0
            right_turn = 0
            above = 0
            right = 0
            below = 0
            left = 0
            
            snake_coord = snake.get_coords()
            apple_coord = apple.get_coordinates()
        
            while True:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        with open("best.pickle", "wb") as f:
                            pickle.dump(Game.best_genome, f)
                        
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            Game.paused()

                dis.fill(Game.background_color)

                #relative food position
                above, right, below, left, same_x, same_y = snake.food_prox(apple_coord[0], apple_coord[1])
                #relative danger
                danger_ahead, danger_right, danger_left = snake.danger_dir()

                max_width_diff = window_width-snake.width
                max_height_diff = window_height-snake.height
                max_dist = np.sqrt(max_height_diff**2+max_width_diff**2) # max distance between the apple and the snake happens when each of them is at opposite corners
                
                dist = np.linalg.norm(snake_coord-apple_coord)
                normalized_dist = normalizer(dist, max_dist, 0)

                abs_x_dist = np.abs(snake_coord[0] - apple_coord[0]) 
                abs_y_dist = np.abs(snake_coord[1] - apple_coord[1]) 
                normalized_abs_x_dist = normalizer(abs_x_dist, max_height_diff, 0)
                normalized_abs_y_dist = normalizer(abs_y_dist, max_width_diff, 0)

                # the inputs are: the location of the apple relative to the snake's head,
                # whether the snake is moving towards the apple,
                # in what direction is there danger (risk of losing),
                # the distance between the snake's head and the apple's location,
                # the absolute distance between the snake's head x coordinate and apple's x coordiante,
                # the absolute distance between the snake's head y coordinate and apple's y coordiante.
                # these inputs are normalized to be between 0 and 1.
                normalized_inputs = (above, right, below, left, same_x, same_y, 
                                    snake.move_dir(apple_coord[0], apple_coord[1]),
                                    danger_ahead, danger_right, danger_left, normalized_dist, 
                                    normalized_abs_x_dist, normalized_abs_y_dist)
                
                output = net.activate(normalized_inputs)
                index = np.argmax(output)
                
                if index == 0:
                    snake.rotate(90)
                    left_turn += 1 
                    right_turn = 0
                    num_left += 1
                elif index == 1:
                    snake.rotate(-90)
                    right_turn += 1
                    left_turn = 0
                    num_right += 1
                else:
                    right_turn, left_turn = 0, 0

                snake.move()
                snake_coord = snake.get_coords()

                if np.array_equal(snake_coord, apple_coord):
                    # rewarding the snake for eating an apple.
                    genome.fitness += 100
                    
                    apple.eaten()
                    apple_coord = apple.get_coordinates()

                    snake.snake_length += 1
                    timer = 0

                    while list(apple_coord) in snake.snake_list:
                        apple.new_coordinates()
                        apple_coord = apple.get_coordinates()

                elif snake.snake_length <= 15:
                    # rewarding the snake for being alive (this is awarded until the snake has eaten 15 apples).
                    genome.fitness += 2/fps
                    timer += 1/fps
                
                if snake.move_dir(apple_coord[0], apple_coord[1]):
                    #rewarding the snake for going towards the apple.
                    genome.fitness += (400/fps)/(np.linalg.norm(snake_coord-apple_coord) + 1)

                if snake_coord[0] < 0 or snake_coord[0] > window_height-snake.height or snake_coord[1] < 0 \
                    or snake_coord[1] > window_width-snake.width:
                    #punishing the snake for losing.
                    genome.fitness -= 10
                    if snake.snake_length > Game.neat_highest_score:
                        Game.neat_highest_score = snake.snake_length
                        Game.best_genome = genome
                    break

                if right_turn > 4 or left_turn > 4:
                    # punishing the snake for rotating around itself.
                    genome.fitness -= 50
                    if snake.snake_length > Game.neat_highest_score:
                        Game.neat_highest_score = snake.snake_length
                        Game.best_genome = genome
                    break
                
                if timer > 20:
                    # punishing the snake for taking to long to eat an apple.
                    genome.fitness -= 50
                    if snake.snake_length > Game.neat_highest_score:
                        Game.neat_highest_score = snake.snake_length
                        Game.best_genome = genome
                    break

                
                if snake.snake_growing():
                    # punishing the snake for colliding with its own body.
                    genome.fitness -= 50
                    if snake.snake_length > Game.neat_highest_score:
                        Game.neat_highest_score = snake.snake_length
                        Game.best_genome = genome
                    break

                snake.render(dis, Game.snake_color)
                if len(snake.snake_list) > snake.snake_length:
                    snake.snake_list.pop(0)

                if snake.snake_x_speed == 20:
                    pygame.draw.circle(dis,black,(snake.snake_x+14,snake.snake_y+6),4)
                    pygame.draw.circle(dis,black,(snake.snake_x+14,snake.snake_y+14),4)
                elif snake.snake_x_speed == -20:
                    pygame.draw.circle(dis,black,(snake.snake_x+6,snake.snake_y+6),4)
                    pygame.draw.circle(dis,black,(snake.snake_x+6,snake.snake_y+14),4) 
                elif snake.snake_y_speed == 20:
                    pygame.draw.circle(dis,black,(snake.snake_x+6,snake.snake_y+14),4)
                    pygame.draw.circle(dis,black,(snake.snake_x+14,snake.snake_y+14),4)
                elif snake.snake_y_speed == -20:
                    pygame.draw.circle(dis,black,(snake.snake_x+6,snake.snake_y+6),4)
                    pygame.draw.circle(dis,black,(snake.snake_x+14,snake.snake_y+6),4)

                dis.blit(apple_image, apple_coord)

                font = pygame.font.Font(pygame.font.match_font("Times New Roman"), 40)
                text_gen = font.render("Generation: " + str(Game.gen), True, Game.snake_color)
                dis.blit(text_gen, (0, 0))
                text_max = font.render("Max score: " + str(Game.neat_highest_score), True, Game.snake_color)
                dis.blit(text_max, (360, 0))
                
                pygame.display.flip()
                clock.tick(fps)

        Game.high_scores.append(Game.neat_highest_score)
        Game.left_turns.append(num_left)
        Game.right_turns.append(num_right)


    @staticmethod
    def  AI_run(config):
        c = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config)
        
        # pop = neat.Checkpointer.restore_checkpoint("neat-checkpoint-54")
        pop = neat.Population(c)
        pop.add_reporter(neat.StdOutReporter(True))
        #stats = neat.StatisticsReporter()
        #pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(5))
        winner = pop.run(Game.AI_eval)

        print('\nBest genome:\n{!s}'.format(winner))
        
                            
    @staticmethod                        
    def game():
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                font = pygame.font.Font(pygame.font.match_font("Times New Roman"), 50)
                text = font.render("Press Space to start", True, orangish)
                dis.fill(Game.background_color)
                dis.blit(background, (44, 0))
                dis.blit(text, (110, 500))
                if event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        Game.main_menu()
            pygame.display.flip()
            

    @staticmethod
    def game_opening():
        dis.fill(Game.background_color)

        clock.tick(1)
        font = pygame.font.SysFont(None, 500)
        text = font.render("3", True, black)
        dis.blit(text, (200, 150))
        pygame.display.flip()
        clock.tick(1)
        dis.fill(Game.background_color)
        text = font.render("2", True, black)
        dis.blit(text, (200, 150))
        pygame.display.flip()
        clock.tick(1)
        dis.fill(Game.background_color)
        text = font.render("1", True, black)
        dis.blit(text, (200, 150))
        pygame.display.flip()
        clock.tick(1)
        dis.fill(Game.background_color)
        text = font.render("GO", True, black)
        dis.blit(text, (40, 150))
        pygame.display.flip()
        clock.tick(1)
        Game.gameloop()


    @staticmethod
    def paused():
        loop = 1

        dis.fill(Game.background_color)
        largeText=pygame.font.Font(pygame.font.match_font("Times New Roman"), 50)
        text_pause = largeText.render('Paused', True, Game.snake_color)
        text_return = largeText.render("Press Esc to return to menu", True, Game.snake_color)
        dis.blit(text_pause, (220, 220))
        dis.blit(text_return, (25, 300))
        pygame.display.flip()
    
        while loop:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        Game.main_menu()
                        loop=0

    @staticmethod
    def game_over_screen(score, number_chili=None):
        pygame.mixer.music.load("sound effects/lose.wav")
        pygame.mixer.music.play()
        score += 3*number_chili
        
        while True:
            dis.fill(Game.background_color)
            if score > Game.highest_score:
                Game.highest_score = score

            font1 = pygame.font.Font(pygame.font.match_font("Times New Roman"), 60)
            font2 = pygame.font.Font(pygame.font.match_font("Times New Roman"), 35)
            text1 = font1.render('Game Over', True, red)
            dis.blit(text1, (160, 240))
            text2 = font1.render("Your score: " + str(score), True, Game.snake_color)
            dis.blit(text2, (140, 120))
            text3 = font2.render('Press Space to play again', True, Game.snake_color)
            dis.blit(text3, (120, 360))
            text4 = font1.render("Highest score: " + str(Game.highest_score), True, Game.snake_color)
            dis.blit(text4, (100, 50))
            text5 = font2.render("Press Esc to return to menu", True, Game.snake_color)
            dis.blit(text5, (105, 450))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == K_SPACE:
                        Game.game_opening()
                        Game.gameloop()
                    if event.key == K_ESCAPE:
                        Game.main_menu()
        
    @staticmethod
    def gameloop():   
        fps = 10
        apple = Consumable("sound effects/munchingfood.mp3") 
        ice = Consumable("sound effects/ice-cracking-01.mp3")
        chili = Consumable("sound effects/mixkit-service-bell-double-ding-588.wav")
        ice_spawn = True
        chili_spawn = True
        keep_ice = False
        keep_chili = False
        number_chili = 0
        snake = Snake()
        snake_coords = snake.get_coords()
        apple_coords = apple.get_coordinates()
        chili_coords = None
        ice_coords = None

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        Game.paused()
                    elif event.key == K_LEFT and snake.snake_x_speed != 20:
                        snake.snake_x_speed = -20
                        snake.snake_y_speed = 0
                    elif event.key == K_RIGHT and snake.snake_x_speed != -20:
                        snake.snake_x_speed = 20
                        snake.snake_y_speed = 0
                    elif event.key == K_DOWN and snake.snake_y_speed != -20:
                        snake.snake_y_speed = 20
                        snake.snake_x_speed = 0
                    elif event.key == K_UP and snake.snake_y_speed != 20:
                        snake.snake_y_speed = -20
                        snake.snake_x_speed = 0
            snake.move()
            snake_coords = snake.get_coords()

            if snake_coords[0] < 0 or snake_coords[0] > window_height-snake.height or snake_coords[1] < 0 \
                or snake_coords[1] > window_width-snake.width:
                Game.game_over_screen(score, number_chili)
                   
            if np.array_equal(snake_coords, apple_coords):
                fps += 0.5
                apple.eaten()
                apple_coords = apple.get_coordinates()

                while (list(apple_coords) in snake.snake_list) or np.array_equal(apple_coords, ice_coords) \
                    or np.array_equal(apple_coords, chili_coords):
                    apple.new_coordinates()
                    apple_coords = apple.get_coordinates()

                snake.snake_length += 1
                ice_spawn = True
                chili_spawn = True

            score = snake.snake_length
            if snake.snake_length >= 10 and snake.snake_length % 5 == 0 and ice_spawn:
                ice_coords = ice.get_coordinates()
                if np.array_equal(snake_coords, ice_coords):
                    fps -= 2
                    ice.eaten()
                    ice_coords = ice.get_coordinates()

                    while (list(ice_coords) in snake.snake_list) or np.array_equal(apple_coords, ice_coords) \
                        or np.array_equal(ice_coords, chili_coords):
                        ice.new_coordinates()
                        ice_coords = ice.get_coordinates()
                    
                    ice_spawn = False
                    keep_ice = False

            elif keep_ice:
                ice_coords = ice.get_coordinates()
                if np.array_equal(snake_coords, ice_coords):
                    fps -= 2
                    ice.eaten()
                    ice_coords = ice.get_coordinates()
                    
                    while (list(ice_coords) in snake.snake_list) or np.array_equal(apple_coords, ice_coords) \
                        or np.array_equal(ice_coords, chili_coords):
                        ice.new_coordinates()
                        ice_coords = ice.get_coordinates()

                    ice_spawn = False
                    keep_ice = False

            if snake.snake_length >= 10 and snake.snake_length % 4 == 2 and chili_spawn:
                chili_coords = chili.get_coordinates()
                if np.array_equal(snake_coords, chili_coords):
                    fps += 3
                    chili.eaten()
                    chili_coords = chili.get_coordinates()
                    number_chili += 1
                    
                    while (list(chili_coords) in snake.snake_list) or np.array_equal(apple_coords, chili_coords) \
                        or np.array_equal(ice_coords, chili_coords):
                        chili.new_coordinates()
                        chili_coords = chili.get_coordinates()

                    chili_spawn = False
                    keep_chili = False
                    
            elif keep_chili:
                chili_coords = chili.get_coordinates()
                if np.array_equal(snake_coords, chili_coords):
                    fps += 3
                    chili.eaten()
                    chili_coords = chili.get_coordinates()
                    number_chili += 1
                    
                    while (list(chili_coords) in snake.snake_list) or np.array_equal(apple_coords, chili_coords) \
                        or np.array_equal(ice_coords, chili_coords):
                        chili.new_coordinates()
                        chili_coords = chili.get_coordinates()

                    chili_spawn = False
                    keep_chili = False
                    
            dis.fill(Game.background_color)

            if snake.snake_growing():
                Game.game_over_screen(score, number_chili)
            
            snake.render(dis, Game.snake_color)
            if len(snake.snake_list) > snake.snake_length:
                snake.snake_list.pop(0)

            if snake.snake_x_speed == 20:
                pygame.draw.circle(dis,black,(snake.snake_x+14,snake.snake_y+6),4)
                pygame.draw.circle(dis,black,(snake.snake_x+14,snake.snake_y+14),4)
            elif snake.snake_x_speed == -20:
                pygame.draw.circle(dis,black,(snake.snake_x+6,snake.snake_y+6),4)
                pygame.draw.circle(dis,black,(snake.snake_x+6,snake.snake_y+14),4) 
            elif snake.snake_y_speed == 20:
                pygame.draw.circle(dis,black,(snake.snake_x+6,snake.snake_y+14),4)
                pygame.draw.circle(dis,black,(snake.snake_x+14,snake.snake_y+14),4)
            elif snake.snake_y_speed == -20:
                pygame.draw.circle(dis,black,(snake.snake_x+6,snake.snake_y+6),4)
                pygame.draw.circle(dis,black,(snake.snake_x+14,snake.snake_y+6),4)     
                
            dis.blit(apple_image, apple_coords)
            if snake.snake_length >= 10 and snake.snake_length % 5 == 0 and ice_spawn:
                dis.blit(ice_image, ice_coords)
                keep_ice = True
            elif keep_ice:
                dis.blit(ice_image, ice_coords)
            if snake.snake_length >= 10 and snake.snake_length % 4 == 2 and chili_spawn:
                dis.blit(chili_image, chili_coords)
                keep_chili = True
            elif keep_chili:
                dis.blit(chili_image, chili_coords)

            pygame.display.flip()
            clock.tick(fps)


if __name__ == '__main__':
    Game.game()
