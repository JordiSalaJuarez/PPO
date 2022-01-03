from encoder import Encoder
from impala import ImpalaModel
from policy import Policy
from utils import make_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
from logger import CSVOutputFormat
from pathlib import Path
from datetime import datetime
import json
import imageio
from augmentation import RandomConvolution
from kornia.augmentation import RandomCrop, ColorJitter, RandomVerticalFlip, RandomHorizontalFlip, RandomGaussianNoise
import logging
logging.basicConfig(level=logging.WARNING)
import uuid

from argparse import ArgumentParser


DEFAULT_ARGS = {
    "total_steps": 8_000_000,
    "num_envs": 32,
    "num_levels": 500,
    "num_steps": 256,
    "num_epochs": 3,
    "n_features": 256,
    "batch_size": 512,
    "eps": .2,
    "grad_eps": .5,
    "value_coef": .1,
    "entropy_coef": .01,
    "env_name": "coinrun",
}

# Seeds used for the enviroments
SEED_TRAIN = 0
SEED_EVAL = 1

# Properties for the image augmentation
SHAPE_CROP = (40,40)
VIDEO_RECORD_RATE = 1_000_000

def parse_args() -> "dict[str, int | float | bool | str]" :
    parser = ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=DEFAULT_ARGS["total_steps"], 
        help="Amount of steps for the training")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_ARGS["num_envs"],
        help="Number of enviroments running the game")
    parser.add_argument("--num_levels", type=int, default=DEFAULT_ARGS["num_levels"],
        help="Number of levels used for the train")
    parser.add_argument("--num_steps", type=int, default=DEFAULT_ARGS["num_steps"],
        help="Number of steps collected from a game")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_ARGS["num_epochs"],
        help="Number of epochs used for training")
    parser.add_argument("--n_features", type=int, default=DEFAULT_ARGS["n_features"],
        help="Number of features used for training (encoder's output size)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_ARGS["batch_size"],
        help="Size of the training batch")
    parser.add_argument("--eps", type=float, default=DEFAULT_ARGS["eps"],
        help="Epsilong value of clipping function")
    parser.add_argument("--grad_eps", type=float, default=DEFAULT_ARGS["grad_eps"],
        help="Epsilong value of clipping function over gradients")
    parser.add_argument("--value_coef", type=float, default=DEFAULT_ARGS["value_coef"],
        help="Coefficient in value objective function")
    parser.add_argument("--entropy_coef", type=float, default=DEFAULT_ARGS["entropy_coef"],
        help="Coefficient in policy entropy")
    parser.add_argument("--use_impala", action="store_true",
        help="Use impala architecture")
    parser.add_argument("--do_crop", action="store_true",
        help="Apply crop image augmentation")
    parser.add_argument("--do_flip", action="store_true",
        help="Apply flip image augmentation")
    parser.add_argument("--do_jitter", action="store_true",
        help="Apply jitter image augmentation")
    parser.add_argument("--same_on_batch", action="store_true",
        help="Same on batch")
    parser.add_argument("--do_gaussian_blur", action="store_true",
        help="Do random gaussian blur")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ARGS["env_name"],
        help="Name of the game used for training")
    parser.add_argument("--use_backgrounds", action="store_true",
        help="Use background for training")

    return parser.parse_args().__dict__

def train(*,     
    total_steps: int,
    num_envs: int,
    num_levels: int,
    num_steps: int,
    num_epochs: int,
    n_features: int,
    batch_size: int,
    eps: float,
    grad_eps: float,
    value_coef: float,
    entropy_coef: float,
    use_impala: bool,
    env_name: str,
    do_crop: bool,
    do_flip: bool,
    do_jitter: bool,
    use_backgrounds: bool,
    same_on_batch: bool,
    do_gaussian_blur: bool,
    ):
    tag = uuid.uuid1()
    start = datetime.now()
    logging.debug('Started Training')
    parameters = {'total_steps': total_steps,
                  'num_envs': num_envs,
                  'num_levels': num_levels,
                  'num_steps': num_steps,
                  'num_epochs': num_epochs,
                  'n_features': n_features,
                  'batch_size': batch_size,
                  'eps': eps,
                  'value_coef': value_coef,
                  'entropy_coef': entropy_coef,
                  'use_impala': use_impala,
                  'env_name': env_name,
                  'do_crop': do_crop,
                  'do_flip': do_flip,
                  'do_jitter': do_jitter,
                  'do_gaussian_blur': do_gaussian_blur,
                  'use_backgrounds': use_backgrounds,
                  'same_on_batch': same_on_batch,}

    assert not(do_crop and use_impala), "Cannot run this configuration"

    base_path = Path("results") / f"{env_name}_{num_levels}_{'impala' if use_impala else 'base'}_{tag}"
    base_path.mkdir(parents=True, exist_ok=True)

    # Save hyperparams in json file, associated to test trough tag
    with open(base_path / "hyperparameters.json", 'w') as outfile:
        json.dump(parameters, outfile, indent=4)

    # Define environment
    # check the utils.py file for info on arguments
    env = make_env(num_envs,env_name=env_name, start_level=SEED_TRAIN*num_levels, num_levels=num_levels, use_backgrounds=use_backgrounds, seed=SEED_TRAIN)
    env_eval = make_env(num_envs,env_name=env_name, start_level=SEED_EVAL*num_levels,num_levels=num_levels, use_backgrounds=use_backgrounds, seed=SEED_EVAL)

    # Define network
    in_channels, w, h = env.observation_space.shape
    if do_crop:
        w, h = SHAPE_CROP
    if use_impala:
        encoder = ImpalaModel(in_channels, n_features)
    else:
        encoder = Encoder(in_channels, n_features, w, h)
    policy = Policy(encoder, n_features, env.action_space.n)
    policy.cuda()

    # Define optimizer
    # these are reasonable values but probably not optimal
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

    # Define temporary storage
    # we use this to collect transitions during each iteration
    storage = Storage(
        (in_channels, w, h),
        num_steps,
        num_envs
    )

    storage_eval = Storage(
        (in_channels, w, h),
        num_steps,
        num_envs
    )

    logger = CSVOutputFormat(base_path / "data.csv")

    def save_clip(path, policy):
        obs = env_eval.reset()
        obs = augmentation(obs)
        frames = []
        total_reward = []

        # Evaluate policy
        policy.eval()
        for _ in range(512):
            # Use policy
            action, log_prob, value = policy.act(obs)

            # Take step in environment
            obs, reward, done, info = env_eval.step(action)
            total_reward.append(torch.Tensor(reward))

            # Render environment and store
            frame = (torch.Tensor(env_eval.render(mode='rgb_array'))*255.).byte()
            frames.append(frame)
            obs = augmentation(obs)

        # Calculate average return
        total_reward = torch.stack(total_reward).sum(0).mean(0)

        # Save frames as video
        frames = torch.stack(frames)
        imageio.mimsave(f'{path}.mp4', frames, fps=25)


    augmentation_set = (
        (do_crop, RandomCrop(SHAPE_CROP, same_on_batch=same_on_batch)),
        (do_flip, RandomHorizontalFlip(same_on_batch=same_on_batch)),
        (do_flip, RandomVerticalFlip(same_on_batch=same_on_batch)),
        (do_gaussian_blur, RandomGaussianNoise(mean=0., std=1., p=1., same_on_batch=same_on_batch)),
        (do_jitter, ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0, same_on_batch=same_on_batch)),
    )
    augmentation = nn.Sequential(*(action for do_action, action in augmentation_set if do_action ))
   

    # Run training
    obs = env.reset()
    obs_eval = env_eval.reset()

    # Apply augmentation
    obs = augmentation(obs)
    obs_eval = augmentation(obs_eval)

    saved_clips = [False] * (total_steps//VIDEO_RECORD_RATE + 1)
    step = 0
    logging.debug('Entering main loop')
    while step < total_steps:
        # Use policy to collect data for num_steps steps
        policy.eval()
        logging.debug('Policy eval')
        for _ in range(num_steps):
            # Use policy
            action, log_prob, value = policy.act(obs)
            action_eval, log_prob_eval, value_eval = policy.act(obs_eval)

            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            next_obs_eval, reward_eval, done_eval, info_eval = env_eval.step(action_eval)

            # Store data
            storage.store(obs, action, reward, done, info, log_prob, value)
            storage_eval.store(obs_eval, action_eval, reward_eval, done_eval, info_eval, log_prob_eval, value_eval)
            
            # Update current observation
            obs = next_obs
            obs_eval = next_obs_eval

            # Apply augmentation
            obs = augmentation(obs)
            obs_eval = augmentation(obs_eval)


        if not saved_clips[step // VIDEO_RECORD_RATE]:
            saved_clips[step // VIDEO_RECORD_RATE] = True
            save_clip(base_path / f"clip_{step//VIDEO_RECORD_RATE}", policy)

        # Add the last observation to collected data
        _, _, value = policy.act(obs)
        storage.store_last(obs, value)
        _, _, value_eval = policy.act(obs_eval)
        storage_eval.store_last(obs_eval, value_eval)

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
                "normalized_reward_train": float(storage.get_reward()),
                "reward_train": float(storage.get_reward(normalized_reward=False)),
                "normalized_reward_test": float(storage_eval.get_reward()),
                "reward_test": float(storage_eval.get_reward(normalized_reward=False)),
                "step": step,
                "time": (datetime.now() - start).total_seconds()
            }
        )
    print('Completed training!')
    torch.save(policy.state_dict, base_path / f'checkpoint.pt')


if __name__ == "__main__":
    args = parse_args()
    train(**args)
