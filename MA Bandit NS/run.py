import numpy as np
import matplotlib.pyplot as plt
from multi_arm_bandit_ns import MABanditNS

actionVals=np.zeros(10)

Bandit1=MABanditNS(actionVals,totalSteps=1000000,method='exponential_weights')
Bandit1.play()
meanReward1=Bandit1.meanRewards

# Bandit2=MABanditNS(actionVals,totalSteps=1000000,method='sample_average')
# Bandit2.play()
# meanReward2=Bandit2.meanRewards

plt.plot(np.arange(Bandit1.totalSteps+1),meanReward1,linewidth=2,color='b',label=f'Weigthed Recency, $\epsilon={Bandit1.epsilon}$')
# plt.plot(np.arange(Bandit2.totalSteps+1),meanReward2,linewidth=2,color='r',label=f'Sampled Average, $\epsilon={Bandit2.epsilon}$')
plt.xscale("log")
plt.xlabel("steps")
plt.ylabel("Average Reward")
plt.legend()
plt.savefig('./results.png',dpi=300)
plt.show()