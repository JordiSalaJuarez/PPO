#%%
import os
import pandas as pd
import seaborn as sns

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_plot(game_name):

    base_paths = list(Path(os.getcwd()).glob(f"results/base_all_games/data_{game_name}*.csv"))
    df_base = pd.read_csv(base_paths[0]).assign(model='base')

    impala_paths = list(Path(os.getcwd()).glob(f"results/impala_all_games/data_{game_name}*.csv"))
    df_impala = pd.read_csv(impala_paths[0]).assign(model='impala')

    # Stack
    df_result = df_base.append(df_impala).reset_index(drop=True)

    # Add rolling means
    rolling_window = 14
    df_result.loc[:, 'rolling_mean'] = df_result.groupby('model')['mean_reward'].rolling(rolling_window).mean().values

    # plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8,6), dpi=144)
    sns.lineplot(data=df_result, x='step', y='mean_reward', hue='model', linewidth=1.2)
    sns.lineplot(data=df_result, x='step', y='rolling_mean', hue='model', palette=['r', 'g'], legend=False, linewidth=1)

    plt.title(game_name.capitalize())
    plt.savefig(f'results/plots/{game_name}.png')

get_plot('starpilot')
get_plot('jumper')
get_plot('chaser')
get_plot('ninja')
