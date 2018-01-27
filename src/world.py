import numpy as np
from params import *
from copy import copy

class world():

    angles = np.arange(0,2*np.pi+delta_angle*np.pi,delta_angle*np.pi)

    def __init__(self, game_mode = "constant_reward",\
    goal_reward = 10,ignore_second_ellipse = True,precision = 1):
        '''
        Setting up the world with random ellipses.
        '''

        self.mode = game_mode
        #Dimensions for three color channels
        self.frame = np.zeros((FRAME_DIM,FRAME_DIM,3))
        self.ignore_second_ellipse = ignore_second_ellipse
        self.goal_reward = goal_reward
        #Random starting state
        self.haxis, self.vaxis,self.haxis2, self.vaxis2 = np.random.random(4)*(FRAME_DIM-MARGIN)/2
        self.set_frame()

        #Initialize target to 10, 10,10,10
        self.opti_haxis, self.opti_vaxis, self.opti_haxis2, self.opti_vaxis2 = [10] * 4

        self.precision = precision

    def set_target(self, vaxis=None,haxis=None, vaxis2 = None, haxis2 = None):
        '''
        For multitarget setting. Allows to reset the target to specific
        location so that reward and game over are calculated correctly.
        All those target axes that are None will not be changed.
        '''
        if vaxis is not None:
            self.opti_vaxis = vaxis
        if haxis is not None:
            self.opti_haxis = haxis
        if haxis2 is not None:
            self.opti_haxis2 = haxis2
        if vaxis2 is not None:
            self.opti_vaxis2 = vaxis2


    def draw_ellipse(self,vaxis,haxis):
        edge = np.array([np.round((haxis*np.cos(angle),vaxis*np.sin(angle))) +\
                              np.array([(FRAME_DIM)/2,(FRAME_DIM)/2])\
                              for angle in self.angles],dtype=int)
        return edge

    def set_frame(self,which = "both"):
        '''
        Redraw ellipse after change.
        Which just ensures that not always all ellipses are redrawn.
        '''
        #Clearing frame



        if which == "both" or which == "first":
            self.frame[:,:,0] *=0
            for point in self.draw_ellipse(self.vaxis,self.haxis):
                self.frame[point[0],point[1],0] = 1

        if which == "both" or which == "second":
            self.frame[:,:,1] *=0
            for point in self.draw_ellipse(self.vaxis2,self.haxis2):
                self.frame[point[0],point[1],1] = 1



    def get_frame(self):
        '''Returns the current ellipse matrix'''
        return copy(self.frame)

    def change_width(self, amount,ellipse=0):
        '''
        Change width by amount pixels.
        If too big or too small, use periodic boundaries.
        ellipse specifies which ellipse shall be changed.
        '''
        if ellipse == 0:
            self.haxis= (self.haxis+amount)%int(FRAME_DIM/2-MARGIN/2)
            self.set_frame(which="first")
        if ellipse == 1:
            self.haxis2= (self.haxis2+amount)%int(FRAME_DIM/2-MARGIN/2)
            self.set_frame(which="second")

    def change_height(self, amount,ellipse=0):
        '''
        Change height by amount pixels.
        If too big or too small, use periodic boundaries.
        '''
        if ellipse == 0:
            self.vaxis= (self.vaxis+amount)%int(FRAME_DIM/2-MARGIN/2)
            self.set_frame(which="first")
        if ellipse == 1:
            self.vaxis2= (self.vaxis2+amount)%int(FRAME_DIM/2-MARGIN/2)
            self.set_frame(which="second")


    def get_reward(self):
        '''
        Getting the reward for different game modes.
        Either constant reward for being close / punishment for being far.
        Or only getting rewards once game is over, otherwise constant.
        TODO Check if still max punishment is necessary for learning.
        TODO For computational efficiency all axes could be held in one vector..
        '''
        if self.mode == "simple_reward" :
            return 0 if not self.game_over() else self.goal_reward

        elif self.ignore_second_ellipse:
            if self.mode == "constant_reward":
                return  -self.measure_distance("first")/(FRAME_DIM/4-MARGIN/4)*10 if not self.game_over() else self.goal_reward
            else:
                raise NotImplementedError

        else:
            if self.mode == "constant_reward":
                #Divide by two to have same scaling in rewards.
                return  -self.measure_distance("both")/(2*(FRAME_DIM/4-MARGIN/4))*10 if not self.game_over() else self.goal_reward

            else:
                raise NotImplementedError


    def distance_per_axis(self,current,target):

        dist = np.abs(current-target)
        if dist > FRAME_DIM/4-MARGIN/4:
            return FRAME_DIM/4-MARGIN/4 - dist%int(FRAME_DIM/4-MARGIN/4)
        return dist


    def measure_distance(self,which = "both"):

        distance = 0
        if which == "both" or which == "first":
            distance += self.distance_per_axis(self.vaxis,self.opti_vaxis)\
                      + self.distance_per_axis(self.haxis,self.opti_haxis)

        if which == "both" or which =="second":
            distance += self.distance_per_axis(self.vaxis2,self.opti_vaxis2)\
                      + self.distance_per_axis(self.haxis2,self.opti_haxis2)

        return distance

    def game_over(self):

        '''Game is over once the goal state is reached to desired precision.
        TODO vectorization would be beneficial here too..
        TODO why not take Euclidean distance?
        '''


        if self.ignore_second_ellipse:
            return int(self.measure_distance("first") < self.precision)

        else :
            return int(self.measure_distance("both") < self.precision)

    def restart(self):
        '''Resetting the world to random state.'''
        self.haxis, self.vaxis,self.haxis2, self.vaxis2 = np.random.random(4)*(FRAME_DIM-MARGIN)/2

        self.set_frame()
