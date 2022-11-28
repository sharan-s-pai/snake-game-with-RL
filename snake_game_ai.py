import pygame
import random
from enum import Enum
import numpy as np
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# RESET: We need a reset the function because we want the agent to reset and start again

# REWARD: Needed for training the agent 

# play(action) -> Must return the direction to move.

# game_iteration
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 90

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    # The agent will be able to reset the game by using this function
    def reset(self):
        # This is the initial state of snake. Agent must always start the game from here.
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y), # Placing the 2nd block behind the head.
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y) # Placing the 3rd block behind 2nd block.
                    ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self,action):
        self.frame_iteration += 1 # Keeps track state time  count (like t,t+1,t+2,...).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # User input is not necessary. It's the agent who decides.
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        reward = 0 # For anything other than dying and eating food, the reward is 0.

        # 3. check if game over
        game_over = False
        
        # We are adding a terminal state check, i.e, as soon as the number of states become 100 times its size then, we close that game (or episode) purposefully
        
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10 # Snake is dead
            return reward,game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food: # If snake eats food (body of snake increases by 1 block)
            self.score += 1
            reward = 10 # Snake has eaten
            self._place_food()
        else: # Else the snake will remain the same size, it would have just moved. So, remove the last block. A new block is added in _move()
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward,game_over, self.score
    
    def is_collision(self,pt=None):
        #  We need to change this function. Rather than just checking for self.head, we need to even keep track of "danger" attributes of the state. So, 
        
        if pt == None:
            pt=self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        # The agent only knows 3 actions straight, turn left, turn right. The game undersands only LEFT,RIGHT,UP,DOWN.

        # ------------------------------------------
        # We are doing the translation here. If you see how the constants are defined, you will understand why our list is clockwise and starts with right.
        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]

        # self.direction keeps track of our current direction always. Trying to get the direction we are in right now. 
        idx = clock_wise.index(self.direction)
        direction = -1
        
        if np.array_equal(action, [1,0,0]): # If our action is to go straight there will be no change in our direction.
            direction = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            # If we are currently in right turning right will lead us down, if we are down, turning right will lead us left. So, r->d->l->u (its clockwise turn)
            next_idx = (idx+1) % 4
            direction = clock_wise[next_idx]
        else: # [0,0,1]
            # If we are currently moving right (-->) turning left will lead us up, if we are up, turning left will lead us left. So, r->u->l->d (its counter-clockwise turn)
            next_idx = (idx-1) % 4
            direction = clock_wise[next_idx]
        # -----------------------------------------
        x = self.head.x
        y = self.head.y
        self.direction = direction
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

if __name__ == '__main__':
    game = SnakeGameAI()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()