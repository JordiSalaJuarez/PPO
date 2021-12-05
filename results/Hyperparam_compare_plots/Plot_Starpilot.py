
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def sliding_mean(x):
    aux = x*0
    for i in range(0,x.shape[0]):
        if i < 10:
            aux[i] = np.mean(x[0:9+i])
        elif i > x.shape[0]-10:
            aux[i] = np.mean(x[i-9:-1])
    
    aux[10:x.shape[0]-9] = np.convolve(x, np.ones(20), 'valid') / 20
            
    return aux


########## PLOTS SAME GAME - DIFFERENT STRUCTURES ##########
mean_reward = []
reward = []
steps = []

fig = plt.figure(figsize=(8,6), dpi=144)


#### PLOT GAME 1 - STARPILOT #####
for path_csv in Path(os.getcwd()).glob("results/Hyperparam_compare_plots/data_starpilot_500*.csv"):
    print(path_csv)
    df = pd.read_csv(path_csv).to_numpy()
#     mean_reward.append(df.iloc[:,0].to_numpy())
#     reward.append(df.loc[:,"reward"].to_numpy())
#     steps.append(df.loc[:,"step"].to_numpy())
# =============================================================================
    mean_reward.append(df[:,0])
    reward.append(df[:,1])
    steps.append(df[:,2])

for path_csv in Path(os.getcwd()).glob("results/Hyperparam_compare_plots/data_starpilot_250*.csv"):
    print(path_csv)
    df = pd.read_csv(path_csv).to_numpy()
#     mean_reward.append(df.iloc[:,0].to_numpy())
#     reward.append(df.loc[:,"reward"].to_numpy())
#     steps.append(df.loc[:,"step"].to_numpy())
# =============================================================================
    mean_reward.append(df[:,2])
    reward.append(df[:,2])
    steps.append(df[:,3])

# Arreglo
# =============================================================================
# steps[2] = steps[2]*8
# steps[3] = steps[3]*8
# 
# steps[4] = steps[4]*4
# steps[5] = steps[5]*4
# =============================================================================

plt.plot(steps[2], sliding_mean(mean_reward[2]), color="#8e7cc3", label="NatureDQN, 1024 steps")
plt.plot(steps[5], sliding_mean(mean_reward[5]),color="#a64d79", label="IMPALA, 1024 steps")

plt.plot(steps[0], sliding_mean(mean_reward[0]), color="#e69138", label="NatureDQN, 512 steps")
plt.plot(steps[1], sliding_mean(mean_reward[1]), color="#3d85c6", label="IMPALA, 512 steps")

plt.plot(steps[4], sliding_mean(mean_reward[4]),color="#da6552", label="NatureDQN, 128 steps")
plt.plot(steps[3], sliding_mean(mean_reward[3]), color="#27A98E", label="IMPALA, 128 steps")

plt.xlabel('steps')
plt.ylabel('mean reward')
plt.title('Comparing performance based on number of steps')
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)

plt.savefig("StarPilot_Structure_Score_Comparison.png", bbox_inches='tight') 
