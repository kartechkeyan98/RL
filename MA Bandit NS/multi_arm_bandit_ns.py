import numpy as np

class MABanditNS():
    '''
    q -> initial true mean values of the rewards
    inc_distro -> elements will be the mean of a guassian which will decide how much change will happen to each element of q
    epsilon -> probability that our agent will explore rather than be greedy
    totalSteps -> total time steps this thing will run for 
    '''

    def __init__(self,q,inc_means=0,epsilon=0.1,totalSteps=10000,method='exponential_weights'):
        
        self.narms=len(q)                   # number of arms
        self.epsilon=epsilon                # epsilon since we will be using an e-greedy method
        self.currentStep=0                  # current time step
        self.armFreq=np.zeros(self.narms)   # vector of arm frequencies
        self.initialValues=q                # true action values (mean), initialized
        self.trueActionValues=q             # this will be updated every time step
        self.Q=np.zeros(self.narms)         # average reward estimates for [0,currentStep-1]
        self.R=0                            # current reward
        self.totalSteps=totalSteps
        self.meanRewards=np.zeros(totalSteps+1) # a list to store how the mean reward progressed with time

        # meant for general case use
        self.inc_means=inc_means            # mean for a guassian which will give you how much to increment true actions values every step
        self.method=method                  # which method to use, sample averages or weghted recency
    
    def update(self,arm):
        sigma_reward=2
        sigma_rew_inc=0.01
        self.currentStep+=1     # increment the current step
        self.armFreq[arm]+=1    # increment frequency of "arm" action

        self.R=np.random.normal(self.trueActionValues[arm],sigma_reward)  # the reward signal mean will change but its variance will be the same

        if(self.method=='sample_average'):
            stepsize=1/(self.armFreq[arm])
        elif(self.method=='exponential_weights'):
            stepsize=0.1
        else:
            print("Enter Valid Method")
            return

        # update overall mean reward
        self.meanRewards[self.currentStep]=self.meanRewards[self.currentStep-1]+(1/(self.currentStep))*(self.R-self.meanRewards[self.currentStep-1])
        # update estimated reward
        self.Q[arm]=self.Q[arm]+stepsize*(self.R-self.Q[arm])
        
        # update reward means
        inc=np.random.normal(self.inc_means,sigma_rew_inc)
        self.trueActionValues=np.add(inc*np.ones((self.narms,)),self.trueActionValues)

    def selectActions(self):

        # usual epsilon greedy approach with some tweeks in the 
        # recursive relation to update Q-values

        p=np.random.rand()

        if(self.currentStep==0) or p<=self.epsilon:
            arm=np.random.choice(self.narms)
        else:
            arm=np.argmax(self.Q)
        
        self.update(arm)
    
    
    def play(self):
        for i in range(self.totalSteps):
            self.selectActions()

    def clearAll(self):
        self.currentStep=0
        self.armFreq=np.zeros(self.narms)
        self.Q=np.zeros(self.narms)
        self.R=0
        self.meanRewards=np.zeros(self.totalSteps+1)
        self.trueActionValues=self.initialValues

        

        
        

