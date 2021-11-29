from encoder import Encoder
from policy import Policy
from utils import make_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
from logger import CSVOutputFormat
from pathlib import Path
from datetime import datetime

import logging
logging.basicConfig(level=logging.WARNING)
import uuid




def train():
    start = datetime.now()
    logging.debug('Started Training')
    # Hyperparameters
    total_steps = 8e6
    num_envs = 32
    num_levels = 10
    num_steps = 256
    num_epochs = 3
    n_features = 256
    batch_size = 512
    eps = .2
    grad_eps = .5
    value_coef = .1
    entropy_coef = .01


    env_name = "starpilot"
    num_levels = 500
    # Define environment
    # check the utils.py file for info on arguments
    env = make_env(num_envs,env_name=env_name,start_level=1, num_levels=num_levels, use_backgrounds=False )
    # print('Observation space:', env.observation_space)
    # print('Action space:', env.action_space.n)

    # Define network
    in_channels = env.observation_space.shape[0]
    encoder = Encoder(in_channels, n_features)
    policy = Policy(encoder, n_features, env.action_space.n)
    policy.cuda()

    # Define optimizer
    # these are reasonable values but probably not optimal
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

    # Define temporary storage
    # we use this to collect transitions during each iteration
    storage = Storage(
        env.observation_space.shape,
        num_steps,
        num_envs
    )


    base_path = "results"
    target_csv = Path(base_path) / f"{env_name}_{num_levels}_{uuid.uudi1()}.csv"
    logger = CSVOutputFormat(target_csv)

    # Run training
    obs = env.reset()
    step = 0
    logging.debug('Entering main loop')
    while step < total_steps:
        # Use policy to collect data for num_steps steps
        policy.eval()
        logging.debug('Policy eval')
        for _ in range(num_steps):
            # Use policy
            action, log_prob, value = policy.act(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)

            # Store data
            storage.store(obs, action, reward, done, info, log_prob, value)
            
            # Update current observation
            obs = next_obs

        # Add the last observation to collected data
        _, _, value = policy.act(obs)
        storage.store_last(obs, value)

        # Compute return and advantage
        storage.compute_return_advantage()

        # Optimize policy
        policy.train()
        logging.debug('Policy train')
        for epoch in range(num_epochs):

            # Iterate over batches of transitions
            generator = storage.get_generator(batch_size)
            for batch in generator:
                b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

                # Get current policy outputs
                new_dist, new_value = policy(b_obs)
                new_log_prob = new_dist.log_prob(b_action)

                # Clipped policy objective
                # calculate surrogates
                ratio = torch.exp(new_log_prob - b_log_prob)
                surrogate_1 = b_advantage * ratio
                surrogate_2 = b_advantage * torch.clamp(ratio, 1-eps, 1+eps)
                pi_loss = -torch.min(surrogate_1, surrogate_2).mean()  # Policy gradient objective, also L^{PG} or PG loss

                # Clipped value function objective
                value_loss_unclipped = (new_value - b_returns)**2
                values_clipped = b_value + torch.clamp(new_value - b_value, -eps, eps)
                value_loss_clipped = (values_clipped - b_returns)**2
                value_loss =  0.5 * torch.mean(torch.max(value_loss_clipped, value_loss_unclipped))

                # Entropy loss
                entropy_loss = new_dist.entropy().mean()

                # Backpropagate losses
                loss = pi_loss + value_coef*value_loss - entropy_coef*entropy_loss # as defined at https://github.com/DarylRodrigo/rl_lib/blob/f165aabb328cb5c798360640fcef58792a72ae8a/PPO/PPO.py#L97
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

                # Update policy
                optimizer.step()
                optimizer.zero_grad()

        # Update stats
        step += num_envs * num_steps
        print(f'Step: {step}\tMean reward: {storage.get_reward()}')
        logger.writekvs(
            {
                "mean_reward": float(storage.get_reward()),
                "reward": float(storage.get_reward(normalized_reward=False)),
                "step": step,
                "time": (datetime.now() - start).total_seconds()

            }
        )
    print('Completed training!')
    torch.save(policy.state_dict, Path(base_path) / 'checkpoint{env_name}_{num_levels}_run{i}.pt')
train()
