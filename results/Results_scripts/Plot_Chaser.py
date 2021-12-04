
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



#### PLOT GAME 1 - STARPILOT #####
for path_csv in Path(os.getcwd()).glob("data_chaser*.csv"):
    print(path_csv)
    df = pd.read_csv(path_csv).to_numpy()
#     mean_reward.append(df.iloc[:,0].to_numpy())
#     reward.append(df.loc[:,"reward"].to_numpy())
#     steps.append(df.loc[:,"step"].to_numpy())
# =============================================================================
    mean_reward.append(df[:,0])
    reward.append(df[:,1])
    steps.append(df[:,2])


# Standard structure
l1 = sns_plot = sns.lineplot(x=steps[0], y=mean_reward[0])
l2 = sns_plot = sns.lineplot(x=steps[0], y=sliding_mean(mean_reward[0]))
l3 = sns_plot = sns.lineplot(x=steps[0], y=reward[0])
l4 = sns_plot = sns.lineplot(x=steps[0], y=sliding_mean(reward[0]))

# IMPALA structure
# =============================================================================
# l5 = sns_plot = sns.lineplot(x=steps[0], y=mean_reward[1])
# l6 = sns_plot = sns.lineplot(x=steps[0], y=sliding_mean(mean_reward[1]))
# l7 = sns_plot = sns.lineplot(x=steps[0], y=reward[1])
# l8 = sns_plot = sns.lineplot(x=steps[0], y=sliding_mean(reward[1]))
# 
# =============================================================================

# IMAGE AUGMENTATION 
# =============================================================================
# l9 = sns_plot = sns.lineplot(x=steps[0], y=mean_reward[2])
# l10 = sns_plot = sns.lineplot(x=steps[0], y=sliding_mean(mean_reward[2]))
# l11 = sns_plot = sns.lineplot(x=steps[0], y=reward[2])
# l12 = sns_plot = sns.lineplot(x=steps[0], y=sliding_mean(reward[2]))
# 
# =============================================================================

sns_plot.set_xlabel('Steps')
sns_plot.set_ylabel('Score')
sns_plot.set_title('Evolution of the score for Chaser')

fig = sns_plot.get_figure()
plt.legend(labels=["Mean reward","MA-Mean reward", "Reward","MA-Reward",
                   #"Mean reward - IMPALA","MA-Mean reward - IMAPALA", "Reward - IMPALA","MA-Reward - IMPALA",
                   #"Mean reward - Augmentation","MA-Mean reward - Augmentation", "Reward - Augmentation","MA-Reward - Augmentation",
                   ], loc = 2, bbox_to_anchor = (1,1))

fig.savefig("Chaser_Structure_Score_Comparison.png") 