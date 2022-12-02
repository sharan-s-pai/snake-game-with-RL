import torch
import random
import numpy as np
from collections import deque # You can pop/push elements only to the beginning/end of queue.
import os

# To give agent complete knowledge of the environment. We need to first get 
from snake_game_ai import SnakeGameAI,Direction,Point,BLOCK_SIZE

from model import Linear_QNet,QTrainer
from helper import plot

MAX_MEMORY=100_000
BATCH = 1000
ALPHA = 0.001

class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0 # Control of randomness
        self.gamma = 0.90 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # As soon as we reach maxlen, it will start popping out element present in the front of the memory.

        # TODO: model, trainer.
        # if not os.path.exists(os.path.join('./model','model.pth')):
        # self.model = Linear_QNet(11,300,3)
        # else:
        self.model = torch.load(os.path.join('./model','model.pth'))
        self.trainer = QTrainer(self.model,lr=ALPHA,gamma=self.gamma)
    
    def get_state(self,game:SnakeGameAI):
        head = game.snake[0]
        # We are trying to extract points and direction to fill the array of 11 elements that will describe a state.
        """
        state = [
            danger_straight: Checks if we go straight, is it dangerous. (boolean)
            danger_left: Checks if we turn left, is it dangerous. (boolean)
            danger_right: Checks if we turn right, is it dangerous. (boolean)

            direction_left: Checks if current direction is left (boolean)
            direction_down: Checks if current direction is down (boolean)
            direction_right: Checks if current direction is right (boolean)
            direction_up: Checks if current direction is up (boolean)

            food_left: Checks if food is on the left of current snake's head position(boolean)
            food_left: Checks if food is below current snake's head position(boolean)
            food_right: Checks if food is on the right of current snake's head position(boolean)
            food_up: Checks if food is above current snake's head position(boolean)
            
        ]
        
        """
        point_l = Point(head.x-BLOCK_SIZE,head.y)
        point_r = Point(head.x+BLOCK_SIZE,head.y)
        point_u = Point(head.x,head.y-BLOCK_SIZE)
        point_d = Point(head.x,head.y+BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or # If we are moving right direction and the immediate straight next position of head is out of the frame/it's own body.
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger to turn right
            (dir_r and game.is_collision(point_d)) or # If we are moving in right direction and the immediate down next position of head is out of frame/part of it's own body
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)),

            # Danger to turn left
            (dir_r and game.is_collision(point_u)) or # If we are moving in right direction and the immediate up next position of head is out of frame/part of it's own body
            (dir_u and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_r)),

            # Move directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < head.x, # Food is left of head position.
            game.food.x > head.x, # Food is right of head position.
            game.food.y < head.y, # Food is below head position.
            game.food.y > head.y # Food is above head position.
        ]
        return np.array(state)

    def remember(self,state,action,reward,next_state,done):
        # Trying to remember the current environment and what next environment state will be.
        self.memory.append((state,action,reward,next_state,done))
        pass

    def train_long_memory(self):
        if len(self.memory) > BATCH:
            mini_sample = random.sample(self.memory,BATCH)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample) # Unpack each element of each tuple in a list and groups them based on their position in tuple. In simple words, row wise grouping is unzipped and column wise grouping is made.
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)
        pass

    def get_action(self,state): # Equivalent to get_move.

        # Here we will use exploration / exploitation: random moves
        self.epsilon = 80 - self.n_games # Here 80 is just some random threshold, till which we may perform random moves. Once epsilon becomes less than 0, random moves are 0.
        final_move = [
                0, # Straight
                0, # Turn right
                0 # Turn left
            ]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2) # Choose some random index.
            final_move[move] = 1 # Set that move to 1
        else:
            # Converting the state into a tensor data-structure.
            state0 = torch.tensor(state,dtype=torch.float32)

            # Our model
            prediction = self.model(state0) # We don't have predict function like in tensorflow. When we use model in pytorch, it directly opens the forward function
            
            # Just like numpy .argmax will return maximum element in the tensor. To convert it back to item 
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    # We will be plotting a graph for scores and mean score.
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 # Best score so far.
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move to be done
        final_move = agent.get_action(state_old)

        reward, game_over, score = game.play_step(final_move)

        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,new_state,game_over)

        agent.remember(state_old,final_move,reward,new_state,game_over)

        if game_over:
            game.reset() # First of all reset. We will start a new cycle

            agent.n_games += 1
            # If the game is over we will try to remember all the moves we have made in this cycle of the game.

            # So, train long memory
            agent.train_long_memory()
            
            if score > record:
                record = score
                if record > 61:
                    agent.model.save()
            
            print('Game ',agent.n_games,'Score: ',score,'Record: ',record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            # This will also be the time where we plot our scores
            


if __name__=='__main__':
    train()