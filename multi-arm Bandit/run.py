import numpy as np
import matplotlib.pyplot as plt
from multi_arm_bandit import MABandit

actionValues=np.array([1,4,2,0,7,1,-1])

e1=0
e2=0.1
e3=0.2
e4=0.3

totalSteps=1000000

# create 4 different bandits

## Greedy Bandit first
Bandit1=MABandit(actionValues,e1,totalSteps)
Bandit1.play()
meanReward1=Bandit1.meanRewards

Bandit2=MABandit(actionValues,e2,totalSteps)
Bandit2.Q=np.zeros(np.size(actionValues))
Bandit2.play()
meanReward2=Bandit2.meanRewards

# Bandit3=MABandit(actionValues,e3,totalSteps)
# Bandit3.play()
# meanReward3=Bandit3.meanRewards

# Bandit4=MABandit(actionValues,e4,totalSteps)
# Bandit4.play()
# meanReward4=Bandit4.meanRewards

plt.plot(np.arange(totalSteps+1),meanReward1,linewidth=2,color='r',label=f'$\epsilon = {e1}, Optimistic Q1 = 5$')
plt.plot(np.arange(totalSteps+1),meanReward2,linewidth=2,color='k',label=f'$\epsilon = {e2}, Realistic Q1=0$')
# plt.plot(np.arange(totalSteps+1),meanReward3,linewidth=2,color='m',label=f'$\epsilon = {e3}$')
# plt.plot(np.arange(totalSteps+1),meanReward4,linewidth=2,color='b',label=f'$\epsilon = {e4}$')
plt.xscale("log")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.savefig('./results.png',dpi=300)
plt.show()