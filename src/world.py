import numpy as np
from params import *
from scipy import ndimage as img
class world():

    def __init__(self,ratio = None, optimal_ratio = 1, game_mode = "abs_length",goal_reward = 10):

        self.mode = game_mode
        self.optimal_ratio = 1

        self.set_target()
        self.goal_reward = goal_reward
        #Random starting state
        self.haxis, self.vaxis = np.random.random(2)*(FRAME_DIM-MARGIN)/2

        self.set_frame()


    def set_target(self, vaxis=10,haxis=10):
        '''
        For multitarget setting. Allows to reset the target to specific
        location so that reward and game over are calculated correctly.
        '''
        self.opti_haxis = haxis
        self.opti_vaxis = vaxis

    def set_frame(self):
        '''
        Redraw ellipse after change.
        '''
        self.frame = np.zeros((FRAME_DIM,FRAME_DIM))

        self.edge = np.array([np.round((self.haxis*np.cos(angle),self.vaxis*np.sin(angle))) +\
                              np.array([(FRAME_DIM)/2,(FRAME_DIM)/2])\
                              for angle in np.arange(0,2*np.pi+delta_angle*np.pi,delta_angle*np.pi)],dtype=int)

        for point in self.edge:
            self.frame[point[0],point[1]] = 1
        self.frame = self.frame.T

    def get_frame(self):
        '''Returns the current ellipse matrix'''
        return self.frame.reshape((FRAME_DIM,FRAME_DIM,1))#img.filters.gaussian_filter(self.frame, 2)

    def change_width(self, amount):
        '''
        Change width by amount pixels.
        If too big or too small, use periodic boundaries.
        '''
        self.haxis= (self.haxis+amount)%int(FRAME_DIM/2-MARGIN/2)
        self.set_frame()

    def change_height(self, amount):
        '''
        Change height by amount pixels.
        If too big or too small, use periodic boundaries.
        '''
        self.vaxis= (self.vaxis+amount)%int(FRAME_DIM/2-MARGIN/2)
        self.set_frame()

    def get_ratio(self):
        '''
        Returns ratio, probably not going to be used anymore.
        '''
        return self.haxis/self.vaxis

    def get_reward(self):
        '''
        Getting the reward for different game modes.
        Either constant reward for being close / punishment for being far.
        Or only getting rewards once game is over, otherwise constant.
        TODO Check if maximal punishment is necessary or even detrimental to learning.
        '''
        if self.mode == "abs_length":
            return -np.max([np.sqrt((self.vaxis-self.opti_vaxis)**2+(self.haxis-self.opti_haxis)**2),5]) \
                if not self.game_over() else self.goal_reward
        if self.mode == "vaxis_only":
            return -np.max([np.sqrt((self.vaxis-self.opti_vaxis)**2),5]) \
                if not self.game_over() else self.goal_reward

        if self.mode == "ratio":
            return -np.max([np.max([self.get_ratio()/self.optimal_ratio,\
                    self.optimal_ratio/self.get_ratio()])**2,5]) \
                    if not self.game_over() else self.goal_reward

        if self.mode == "simple_abs" or self.mode == "simple_vaxis_only":
            return 0 if not self.game_over() else self.goal_reward

        else:
            raise NotImplementedError

    def game_over(self, precision = .05):

        '''Game is over once the goal state is reached to desired precision.'''
        if  "vaxis_only" in self.mode:
            return np.abs(self.vaxis-self.opti_vaxis) < precision

        if self.mode == "abs_length" or self.mode == "simple_abs":
            return np.abs(self.vaxis-self.opti_vaxis) < precision and  np.abs(self.haxis-self.opti_haxis) < precision
        if self.mode == "ratio" :
            return self.get_ratio()/self.optimal_ratio < precision
        else :
            raise NotImplementedError

    def restart(self):
        '''Resetting the world to random state.'''
        self.haxis, self.vaxis = np.random.random(2)*(FRAME_DIM-MARGIN)/2
        self.set_frame()
