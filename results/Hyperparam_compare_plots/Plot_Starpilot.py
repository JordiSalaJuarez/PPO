
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
for path_csv in Path(os.getcwd()).glob("data_starpilot_500*.csv"):
    print(path_csv)
    df = pd.read_csv(path_csv).to_numpy()
#     mean_reward.append(df.iloc[:,0].to_numpy())
#     reward.append(df.loc[:,"reward"].to_numpy())
#     steps.append(df.loc[:,"step"].to_numpy())
# =============================================================================
    mean_reward.append(df[:,0])
    reward.append(df[:,1])
    steps.append(df[:,2])

for path_csv in Path(os.getcwd()).glob("data_starpilot_250*.csv"):
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


# Standard structure
#l1 = sns_plot = sns.lineplot(x=steps[0], y=mean_reward[0])
l2 = sns_plot = sns.lineplot(x=steps[0], y=sliding_mean(mean_reward[0]))
#l3 = sns_plot = sns.lineplot(x=steps[2], y=mean_reward[2])
l4 = sns_plot = sns.lineplot(x=steps[1], y=sliding_mean(mean_reward[1]))
#l5 = sns_plot = sns.lineplot(x=steps[4], y=mean_reward[4])
l6 = sns_plot = sns.lineplot(x=steps[2], y=sliding_mean(mean_reward[2]))

# IMPALA structure
#l1imp = sns_plot = sns.lineplot(x=steps[1], y=mean_reward[1])
l2imp = sns_plot = sns.lineplot(x=steps[3], y=sliding_mean(mean_reward[3]))
#l3imp = sns_plot = sns.lineplot(x=steps[3], y=mean_reward[3])
l4imp = sns_plot = sns.lineplot(x=steps[4], y=sliding_mean(mean_reward[4]))
#l5imp = sns_plot = sns.lineplot(x=steps[5], y=mean_reward[5])
l6imp = sns_plot = sns.lineplot(x=steps[5], y=sliding_mean(mean_reward[5]))


sns_plot.set_xlabel('Steps')
sns_plot.set_ylabel('Score')
sns_plot.set_title('Evolution of the score for StarPilot')

fig = sns_plot.get_figure()
plt.legend([l2, l4, l6, l2imp, l4imp, l6imp], labels=["MA-Mean reward", "MA-Mean reward Impala",
                                                      "MA-Mean reward decrease", "MA-Mean reward decrease Impala",
                                                      "MA-Mean reward increase", "MA-Mean reward increase Impala"
                                                        ], loc = 2, bbox_to_anchor = (1,1))

fig.savefig("StarPilot_Structure_Score_Comparison.png") 
