import torch
import random
import numpy
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 # items in the memory
BATCH_SIZE = 1000
LR = 0.001

HIDDEN_UNITS = 256

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # controll the randomness of moves
        self.gamma = 0.9 # discount rate<1, around 0.8-0.9 often
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if memory exceeded
        # model, trainer
        self.model = Linear_QNet(input_features=11, output_features=3, hidden_units=HIDDEN_UNITS)
        self.trainer = QTrainer(self.model, LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0] # grab the head from the snake
        point_l = Point(head.x-20, head.y) # Creates points l,r,u,d to the head
        point_r = Point(head.x+20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        dir_l = game.direction == Direction.LEFT # checks if the direction is equal to the l,r,u,d
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or # om du rör dig höger och du kolliderar höger 1 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        return np.array(state, dtype=int) #! why not bool?

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # if this exeeds max memory it pops the leftmost item

    def train_long_memory(self): # for each and every step in a game (batch)
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # return a list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample) # basically transpose
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        

    def train_short_memory(self, state, action, reward, next_state, done): # for one step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # in the beginning there will be a lot of exploration <=> randomness, then it will turn to exploitation <=> ai choices
        # this is done via using the epsilon variable
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: # as epsilon goes towards 0, the chance increases
            # in the beginning its 80/200 -> 40% chance of a random move
            move = random.randint(0, 2) # 0,1,2 
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # forward = predict
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_mean_five_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True: # The training loop
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move + get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state = state_old, action = final_move, reward= reward, next_state = state_new, done = done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory / experiance replay and go through the previous games
            # plot the results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()
            print("Game: ", agent.n_games, "  Score: ", score, "  Record: ", record)

            # plot

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/ agent.n_games
            plot_mean_scores.append(mean_score)

            mean_five_score = sum(plot_scores[-20:])/20
            plot_mean_five_scores.append(mean_five_score)

            plot(plot_scores, plot_mean_scores, plot_mean_five_scores)

        

if __name__ == "__main__":
    train()
