import numpy as np
from params import *
from copy import copy
from sys import exit
import pickle

class world():

    #Angles are not used in-game anymore.
    angles = np.arange(0,2*np.pi+delta_angle*np.pi,delta_angle*np.pi)
    #Number that constantly comes up.
    max_length = int((FRAME_DIM-MARGIN)/2)


    def __init__(self, game_mode = "constant_reward",\
    goal_reward = 10,ignore_second = True,ignore_first = False,\
    precision = 1, first_in_second = False, both_in_first = False):
        '''
        Setting up the world with random ellipses.
        Keywords:
            game_mode :     "constant_reward" or "simple_reward"
                            The former gives out non-zero rewards after every action,
                            the latter only once the goal is reached.

            goal_reward:    Magnitude of reward returned once goal is reached.

            ignore_second:  If set to True, rewards will only be given according
                            to the state of the first geometric form.
                            I.e. for constant reward only the distance to the
                            correct state of the first geometric form is rewarded,
                            as is the goal state.
            ignore_first:   Same as ignore_second, now first form is ignored
                            and second rewarded. They cannot be both set to True.

            precision:      Goal is reached once distance to goal is smaller than
                            precision. Relates to the number of steps that are
                            still to be taken in order to reach the goal state.

            first_in_second: This mode copies the form from the first color
                            channel also into the second color channel, in order
                            to make teh distinction between first and second
                            not as clear as in the seperated color channel case.

            both_in_first:  This mode will only make use of one color channel.
                            Instead of using two ellipses that would now
                            be undistinguishable, the second form will be a
                            rectangle. This should be the most difficult of the
                            tasks until now.

        '''

        #If both were set to True, no rewards could ever be earnt.
        if ignore_first and ignore_second:
            print("You cannot ignore both ellipses... exiting.")
            exit(1)

        #Load ellipses for more efficient training and avoiding to redraw
        #the images at each time step.
        with open("ellipses.pkl","rb") as file:
            self.all_ellipses = pickle.load(file)



        self.first_in_second = first_in_second
        self.both_in_first = both_in_first
        self.mode = game_mode

        #If both forms will be in the first channel only,
        #no need to use more than one channel.
        if self.both_in_first:
            self.frame = np.zeros((FRAME_DIM,FRAME_DIM,1))
        #three channels for other cases, although 2 are used maximally.
        #conv2d layer only configured to take either 1,3, or 4 channels..
        #lazy solution
        else:
            self.frame = np.zeros((FRAME_DIM,FRAME_DIM,3))


        self.ignore_second = ignore_second
        self.ignore_first = ignore_first
        self.goal_reward = goal_reward

        #Random starting state
        self.restart()

        #Initialize target to 10, 10,10,10, usually a target is set during the
        #training though.
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
        '''
        Returns the matrix for an ellipse as specified by vaxis and haxis.
        This was used to create all the ellipses that are loaded into
        all_ellipses.
        '''
        #setting up the matrix
        ellipse = np.zeros((FRAME_DIM,FRAME_DIM))
        #choosing the points that belong to the ellipse.
        edge = np.array([np.round((haxis*np.cos(angle),vaxis*np.sin(angle))) +\
                              np.array([(FRAME_DIM)/2,(FRAME_DIM)/2])\
                              for angle in self.angles],dtype=int)

        #increase efficiency for the loop below.
        edge = np.unique(edge,axis=0)
        #filling the matrix
        for point in edge:
                ellipse[point[0],point[1]] = 1
        return ellipse

    def draw_square(self,vaxis,haxis):
        '''
        Drawing a square is comparably simple, therefore for now not pickled.
        '''
        center = int(FRAME_DIM/2)
        square = np.zeros((FRAME_DIM,FRAME_DIM))
        square[center+vaxis,center-haxis:center+haxis+1] = 1
        square[center-vaxis,center-haxis:center+haxis+1] = 1
        square[center-vaxis:center+vaxis+1,center+haxis] = 1
        square[center-vaxis:center+vaxis+1,center-haxis] = 1

        return square

    def set_frame(self,which = "both",load_frame=True):
        '''
        Redraw ellipse after change.
        Which just ensures that not always all ellipses are redrawn.
        '''

        if load_frame:

            if self.both_in_first:
                self.frame[:,:,0] *=0
                #choosing the appropriate ellipse from the loaded ellipses.
                self.frame[:,:,0] += self.all_ellipses[(self.vaxis-1)*(self.max_length)+(self.haxis-1)]
                self.frame[:,:,0] += self.draw_square(self.vaxis2,self.haxis2)
                self.frame[:,:,0] = np.clip(self.frame[:,:,0],0,1)
                return

            if which == "first" or which == "both":
                self.frame[:,:,0] *=0
                self.frame[:,:,0] += self.all_ellipses[(self.vaxis-1)*(self.max_length)+(self.haxis-1)]

            if which == "second" or which == "both":
                self.frame[:,:,1] *=0
                self.frame[:,:,1] += self.all_ellipses[(self.vaxis2-1)*(self.max_length)+(self.haxis2-1)]

            if self.first_in_second:
                self.frame[:,:,1] = np.clip(self.frame[:,:,0]+self.frame[:,:,1],0,1)

            return


    def get_frame(self):
        '''Returns the current ellipse matrix.'''
        return copy(self.frame)

    def change_width(self, amount,ellipse=0):
        '''
        Change width by amount of pixels.
        If too big or too small, use periodic boundaries.
        ellipse specifies which ellipse shall be changed.
        '''
        if ellipse == 0:
            self.haxis= (self.haxis+amount)%self.max_length
            self.set_frame(which="first")
        if ellipse == 1:
            self.haxis2= (self.haxis2+amount)%self.max_length
            self.set_frame(which="second")

    def change_height(self, amount,ellipse=0):
        '''
        Change height by amount of pixels.
        If too big or too small, use periodic boundaries.
        '''
        if ellipse == 0:
            self.vaxis= (self.vaxis+amount)%self.max_length
            self.set_frame(which="first")
        if ellipse == 1:
            self.vaxis2= (self.vaxis2+amount)%self.max_length
            self.set_frame(which="second")


    def get_reward(self):
        '''
        Getting the reward for different game modes.
        Either constant reward for being close / punishment for being far.
        Or only getting rewards once game is over, otherwise 0.
        '''
        if self.mode == "simple_reward" :
            return 0 if not self.game_over() else self.goal_reward

        elif self.ignore_second:
            if self.mode == "constant_reward":
                return  -self.measure_distance("first")/(self.max_length/2)*10 \
                if not self.game_over() else self.goal_reward
            else:
                raise NotImplementedError

        elif self.ignore_first:
            if self.mode == "constant_reward":
                return  -self.measure_distance("second")/(self.max_length/2)*10 \
                if not self.game_over() else self.goal_reward
            else:
                raise NotImplementedError

        else:
            if self.mode == "constant_reward":
                #Divide by two to have same scaling in rewards.
                return  -self.measure_distance("both")/(self.max_length)*10 \
                if not self.game_over() else self.goal_reward

            else:
                raise NotImplementedError


    def distance_per_axis(self,current,target):
        '''
        Helper function to calculate distance from goal.
        '''
        dist = np.abs(current-target)
        if dist > self.max_length/2:
            return self.max_length/2 - dist%int(self.max_length/2)
        return dist


    def measure_distance(self,which = "both"):

        '''
        Distance measure to determine distance from goal state,
        depending on the game mode.
        '''
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
        '''


        if self.ignore_second:
            return int(self.measure_distance("first") < self.precision)
        elif self.ignore_first:
            return int(self.measure_distance("second") < self.precision)

        else :
            return int(self.measure_distance("both")/2 < self.precision)

    def restart(self):
        '''Resetting the world to random state.'''
        self.haxis, self.vaxis,self.haxis2, self.vaxis2 = np.random.randint(1,self.max_length +1,size=4 )

        self.set_frame()
