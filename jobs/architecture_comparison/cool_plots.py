import os
import pandas as pd
import seaborn as sns

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_plot():

    def game_name(csv:str):
        name = str(csv)
        return name.split('/')[-1].split('_')[1]

    csv = list(Path(os.getcwd()).glob(f"jobs/architecture_comparison/impala/*.csv"))
    csv.sort()
    plt.gcf().subplots_adjust(bottom=0.15)

    fig, axes = plt.subplots(1, 4, sharex=True, figsize=(20,4))
    rolling_window = 20
    
    for i in range(len(csv)):

        df_result = pd.read_csv(csv[i]).assign(model='NatureDQN')
        df_result.loc[:, 'rolling_mean_train'] = df_result.groupby('model')['normalized_reward_train'].rolling(rolling_window).mean().values
        df_result.loc[:, 'rolling_mean_test'] = df_result.groupby('model')['normalized_reward_test'].rolling(rolling_window).mean().values

        sns.lineplot(ax=axes[i], data=df_result, x='step', y='rolling_mean_train', label='train', legend=0)
        sns.lineplot(ax=axes[i], data=df_result, x='step', y='rolling_mean_test', label='test', legend=0)
        axes[i].set_title(game_name(csv[i]))
        axes[i].set_ylabel('')    
        axes[i].set_xlabel('')

    axes[0].set_ylabel('normalized reward')
    handles, labels = axes[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right') 
    fig.supxlabel('step')

    # plt.show()
    plt.savefig(f'impala.png', bbox_inches='tight')

get_plot()
