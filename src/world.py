import numpy as np
from params import *
from scipy import ndimage as img
class world():

    def __init__(self,ratio = None, optimal_ratio = 1, game_mode = "ratio",goal_reward = 10):

        self.mode = game_mode
        self.optimal_ratio = 1

        self.set_target()
        self.goal_reward = goal_reward
        #ratio represents the ratio between the two main axes, namely horizontal / vertical.

        if ratio is None:
            self.ratio = np.random.random()*(MAX_RATIO-MIN_RATIO)+MIN_RATIO
        else: self.ratio = ratio

        #Initialize horizontal axis randomly, set vertical axis to fulfill ratio requirements.
        #Make sure neither one of the axis will let part of the ellipse be outside the frame by
        #using the ratio to cap the random length. Lengths measured in pixels.
        self.haxis = np.random.random()*(FRAME_DIM-2*MARGIN)/2
        if self.haxis / self.ratio >= (FRAME_DIM/2-MARGIN):
            self.haxis *=self.ratio

        self.vaxis = self.haxis/self.ratio
        #print(self.haxis,self.vaxis)

        self.set_frame()
        self.ax_dict = {"haxis" :self.haxis, "vaxis": self.vaxis}


    def set_target(self, vaxis=10,haxis=10):

        self.opti_haxis = haxis
        self.opti_vaxis = vaxis

    def set_frame(self):

        self.frame = np.zeros((FRAME_DIM,FRAME_DIM))

        self.edge = np.array([np.round((self.haxis*np.cos(angle),self.vaxis*np.sin(angle))) +\
                              np.array([(FRAME_DIM)/2,(FRAME_DIM)/2])\
                              for angle in np.arange(0,2*np.pi+delta_angle*np.pi,delta_angle*np.pi)],dtype=int)

        for point in self.edge:
            self.frame[point[0],point[1]] = 1
        self.frame = self.frame.T

    def get_frame(self):
        return self.frame.reshape((FRAME_DIM,FRAME_DIM,1))#img.filters.gaussian_filter(self.frame, 2)

    def change_width(self, amount):

        if not self.haxis*(1+amount) >= FRAME_DIM/2-MARGIN:
            self.haxis*=(1+amount)
        if self.haxis < 1:
            self.haxis =  FRAME_DIM/2-MARGIN - 1
        self.set_frame()

    def change_height(self, amount):

        if not np.abs(self.vaxis*(1+amount)) >= FRAME_DIM/2-MARGIN:
            self.vaxis*=(1+amount)
        if self.vaxis < 1:
            self.vaxis =  FRAME_DIM/2-MARGIN - 1
        self.set_frame()

    def get_ratio(self):
        return self.haxis/self.vaxis

    def get_reward(self):

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

        if  "vaxis_only" in self.mode:
            return np.abs(self.vaxis-self.opti_vaxis) < precision

        if self.mode == "abs_length" or self.mode == "simple_abs":
            return np.abs(self.vaxis-self.opti_vaxis) < precision and  np.abs(self.haxis-self.opti_haxis) < precision
        if self.mode == "ratio" :
            return self.get_ratio()/self.optimal_ratio < precision
        else :
            raise NotImplementedError

    def restart(self):
        self.__init__(optimal_ratio=self.optimal_ratio,game_mode=self.mode)
