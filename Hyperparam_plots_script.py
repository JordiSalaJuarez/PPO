# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sliding_mean(x):
    aux = x*0
    for i in range(0,x.shape[0]):
        if i < 10:
            aux[i] = np.mean(x[0:9+i])
        elif i > x.shape[0]-10:
            aux[i] = np.mean(x[i-9:-1])
    
    aux[10:x.shape[0]-9] = np.convolve(x, np.ones(20), 'valid') / 20
            
    return aux

def plot_intervalo(steps,reward,color,label):
    
    window = pd.Series(reward).rolling(20)
    q1 = window.quantile(0.25)
    q2 = window.quantile(0.5)
    q3 = window.quantile(0.75)
    plt.fill_between(steps, q1,q3, color=color, alpha=0.2)
    plt.plot(steps, q2, color=color, label=label)

##  CHASER - EPS ###

c1,c2,c3 = sns.color_palette("mako_r", 3)

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Chaser/Base/Standard.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c1, "Cem = 0.2")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Chaser/Base/High.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c2, "Cem = 0.4")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Chaser/Base/Low.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c3, "Cem = 0.05")


plt.xlabel('Steps')
plt.ylabel('Score')
plt.title('Clipping error margin - CHASER')

plt.legend()
plt.savefig("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Billeder/Saved Pictures/Cem_Chaser.png") 
plt.show()


##  CHASER - ENTROPY_COEF ###
df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Chaser/Base/Standard.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c1, "Cem = 0.01")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Chaser/Base/High.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c2, "Cem = 0.02")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Chaser/Base/Low.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c3, "Cem = 0.005")


plt.xlabel('Steps')
plt.ylabel('Score')
plt.title('Entropy coefficient - CHASER')

plt.legend()
plt.savefig("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Billeder/Saved Pictures/E_coef_Chaser.png") 

plt.show()



##  JUMPER - EPS ###

palette = sns.color_palette("mako_r", 5)

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Jumper/Base/Standard.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c1, "Cem = 0.2")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Jumper/Base/High.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c2, "Cem = 0.4")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Jumper/Base/Low.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c3, "Cem = 0.05")

plt.xlabel('Steps')
plt.ylabel('Score')
plt.title('Clipping error margin - Jumper')

plt.legend()
plt.savefig("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Billeder/Saved Pictures/Cem_Jumper.png") 
plt.show()




##  JUMPER - ENTROPY_COEF ###
df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Jumper/Base/Standard.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c1, "Cem = 0.01")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Jumper/Base/High.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c2, "Cem = 0.02")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Jumper/Base/Low.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c3, "Cem = 0.005")

plt.xlabel('Steps')
plt.ylabel('Score')
plt.title('Entropy Coefficient - Jumper')

plt.legend()
plt.savefig("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Billeder/Saved Pictures/E_coef_Jumper.png") 
plt.show()



##  NINJA - EPS ###

palette = sns.color_palette("mako_r", 5)

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Ninja/Base/Standard.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c1, "Cem = 0.2")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Ninja/Base/High.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c2, "Cem = 0.4")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/eps/Ninja/Base/Low.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c3, "Cem = 0.05")

plt.xlabel('Steps')
plt.ylabel('Score')
plt.title('Clipping error margin - Ninja')

plt.legend()
plt.savefig("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Billeder/Saved Pictures/Cem_Ninja.png") 
plt.show()


##  NINJA - ENTROPY_COEF ###
df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Ninja/Base/Standard.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c1, "Cem = 0.01")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Ninja/Base/High.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c2, "Cem = 0.02")

df = pd.read_csv("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Dokumenter/DTU_resources/PPO/jobs/Hyperparam_ordered/entropy_coefs/Ninja/Base/Low.csv")
plot_intervalo(df["step"], df["normalized_reward_train"],c3, "Cem = 0.005")

plt.xlabel('Steps')
plt.ylabel('Score')
plt.title('Entropy coefficient - Ninja')

plt.legend()
plt.savefig("C:/Users/Pau/OneDrive - Danmarks Tekniske Universitet/Billeder/Saved Pictures/E_coef_Ninja.png") 
plt.show()


