import numpy as np

class MABandit():
    # trueActionValues- Actual mean/expected rewards for the k actions
    # k is the number of entries in the trueActionValues
    # epsilon- epsilon probability value for selecting non-greedy approach
    # totalSteps- number of iterations or steps to run the model for

    def __init__(self,trueActionValues,epsilon,totalSteps):
        # number of arms
        self.narms=np.size(trueActionValues)

        # prob of ignoring greedy selection and selecting random arm
        self.epsilon=epsilon

        # what time step you are at
        self.currentStep=0

        # keep track of how many times a particular arm is selected, basically a frequency chart
        self.armFrequency=np.zeros(self.narms)

        # total number of iterations in the run
        self.totalSteps=totalSteps

        self.trueActionValues=trueActionValues

        #vector to store the estimated mean reward for every arm we call it Q
        self.Q=5*np.ones((self.narms,))

        # total reward you have till now- call it R
        self.R=0

        # mean reward for every iteration
        self.meanRewards=np.zeros(totalSteps+1)

    def selectActions(self):
        # draw a real number from uniform(0,1)
        # this number p if <= epsilon, we explore
        # else the usual drill- greedy approach

        p=np.random.rand()

        # select greedy or non-greedy
        if(self.currentStep==0) or (p<=self.epsilon):
            arm=np.random.choice(self.narms)    # non-greedy
        else:
            arm=np.argmax(self.Q)       # greedy
        
        self.currentStep+=1     # current step inc.
        self.armFrequency[arm]+=1  # no.of times the selected arm was taken inc.
        
        # the reward distro for action a_k ~ N(mu_k,2)
        self.R=np.random.normal(self.trueActionValues[arm],2)

        # mean reward for whole run => total reward = mean*k, updated
        self.meanRewards[self.currentStep]=self.meanRewards[self.currentStep-1]+(1/(self.currentStep))*(self.R-self.meanRewards[self.currentStep-1])
        
        # update estimates of mean reward for selected arm
        self.Q[arm]=self.Q[arm]+(1/(self.armFrequency[arm]))*(self.R-self.Q[arm])
    
    # run the simulation
    def play(self):
        for i in range(self.totalSteps):
            self.selectActions()
    
    def clearAll(self):
        self.currentStep=0
        self.armFrequency=np.zeros(self.narms)
        self.Q=np.zeros(self.narms)
        self.R=0
        self.meanRewards=np.zeros(self.totalSteps+1)
    
