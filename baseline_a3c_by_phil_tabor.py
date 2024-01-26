# original code by Phil Tabor https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py
# Code modified by Christian Gurski

# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C
import gymnasium as gym  # stcngurs: switched from gym to gymnasium
import numpy as np
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical

from src.common.early_stopping import EarlyStopping
from src.trainers.a3c_functions import save_gradients

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, target_score=200, run = None,
                 early_stopping_tolerance = 50, dropout: float = 0):  #stcngurs: added params target_score, run, early_stopping_tolerance
        super(ActorCritic, self).__init__()
        self.dropout = dropout
        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []
        self.metrics: dict = {}

        self.agent_data: dict = {}
        self.agents_total_epochs = 0
        self.agents_total_steps = 0

        self.run = run  #stcngurs: added wandb run
        self.early_stopping = EarlyStopping(tolerance=early_stopping_tolerance,
                                            target_score=target_score)  # stcngurs: added early stopping


    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.dropout(F.relu(self.pi1(state)), self.dropout)
        v1 = F.dropout(F.relu(self.v1(state)), self.dropout)

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)

        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        self.metrics["loss"] = float(total_loss.cpu().detach().numpy())
        self.metrics["actor_loss"] = float(actor_loss.cpu().detach().numpy()[-1])
        self.metrics["critic_loss"] = float(critic_loss.cpu().detach().numpy()[-1])
        self.metrics["advantage"] = float((returns - values).cpu().detach().numpy()[-1])

        return total_loss

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = int(dist.sample()[0])

        return action

    def add_agent_metrics(self, data_dict):
        if "agent_id" in data_dict:
            agent_id = data_dict["agent_id"]
            self.agent_data[agent_id] = data_dict


    def calculate_metrics(self):
        all_metrics = []
        all_keys: dict = {}
        mean: dict = {}
        agent_scores: dict = {}
        for agent_id in self.agent_data:
            if len(self.agent_data[agent_id]) > 0:
                agent_data = self.agent_data[agent_id]
                if "metrics" in agent_data:
                    metrics = agent_data["metrics"]
                    all_keys.update(metrics)
                    all_metrics.append(metrics)
                    if "score" in metrics:
                        agent_scores[agent_id] = metrics["score"]

        for metrics in all_metrics:
            for key in all_keys:
                if isinstance(all_keys[key], float):
                    if key not in metrics:
                        metrics[key] = 0
                    mean[key] = 0.0
                if isinstance(all_keys[key], int):
                    if key not in metrics:
                        metrics[key] = 0
                    mean[key] = 0

        for key in mean:
            for metrics in all_metrics:
                if (isinstance(mean[key], float) or isinstance(mean[key], int)) and (
                        isinstance(metrics[key], float) or isinstance(metrics[key], int)):
                    mean[key] += metrics[key]
            if isinstance(mean[key], float) or isinstance(mean[key], int):
                mean[key] = mean[key] / len(all_metrics)

        self.metrics.update(mean)
        self.metrics["global_agents_total_epochs"] = self.agents_total_epochs
        self.metrics["global_agents_total_steps"] = self.agents_total_steps

        """
        agent_ids_sorted = list(agent_scores.keys())
        agent_ids_sorted.sort()
        scores = list(agent_scores.values())
        table = wandb.Table(data=[scores], columns=agent_ids_sorted)
        self.metrics["agent_scores"] = wandb.plot.histogram(table, "agent scores",
                                                                       title="Agents scores")
        """

        if "score" in self.metrics:
            # early stopping watch score
            self.early_stopping(score=self.metrics["score"])  # stcngurs: added early stopping
        self.run.log(self.metrics)


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr, name, global_ep_idx, env_id, N_GAMES, T_MAX, dropout):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma, dropout=dropout)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer
        self.N_GAMES = N_GAMES
        self.T_MAX = T_MAX

    def run(self):
        t_step = 1
        while self.episode_idx.value < self.N_GAMES:
            done = False
            observation = self.env.reset()[0]
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, terminated, truncated, info = self.env.step(action)
                done = terminated
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % self.T_MAX == 0 or done or truncated:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()

                    self.global_actor_critic.calculate_metrics()  # stcngurs: added logging global step
                    self.global_actor_critic.add_agent_metrics({"agent_id": self.name,
                                                                "metrics": self.local_actor_critic.metrics})
                t_step += 1
                self.global_actor_critic.agents_total_steps += 1
                observation = observation_
                self.local_actor_critic.metrics["steps"] = t_step

                if truncated:
                    break

            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score, str(truncated))
            self.local_actor_critic.metrics["score"] = score
            self.local_actor_critic.metrics["finished"] = int(truncated)
            self.global_actor_critic.agents_total_epochs += 1

            if self.global_actor_critic.early_stopping.early_stop:  # stcngurs: added early stopping
                print("early stopping")
                break


if __name__ == '__main__':
    lr = 1e-4
    env_id = 'CartPole-v0'
    target_score = 200
    n_actions = 2
    input_dims = [4]
    N_GAMES = 3000000
    T_MAX = 200  # batchsize
    betas = (0.90, 0.999)  # stcngurs: used default betas instead of: betas=(0.92, 0.999)
    num_workers = 1  # stcngurs: used 15 instead of mp.cpu_count()
    early_stopping_tolerance = 50
    dropout: float = 0.0

    wandb_sweep_config = {
        'name': 'sweep',
        'method': 'grid',  # grid, random, bayes
        'metric': {'goal': 'minimize', 'name': 'loss'},
        'parameters': {
            "lr": {"value": lr},
            "optimizer": {"value": "shared_adam"},
            "env_id": {"value": env_id},
            "max_epochs": {"value": N_GAMES},
            "batch_size": {"value": T_MAX},
            "betas": {"value": betas},
            "num_agents": {"value": num_workers},
            "target_score": {"value": target_score},
            "early_stopping_tolerance": {"value": early_stopping_tolerance}
        }
    }
    run = wandb.init(project="a3c_global_phil_tabor", config=wandb_sweep_config)

    global_actor_critic = ActorCritic(input_dims, n_actions, target_score=target_score, run=run,
                                      early_stopping_tolerance=early_stopping_tolerance, dropout=dropout)
    global_actor_critic.share_memory()

    optim = SharedAdam(global_actor_critic.parameters(), lr=lr,
                       betas=betas)
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                     optim,
                     input_dims,
                     n_actions,
                     gamma=0.99,
                     lr=lr,
                     name=i,
                     global_ep_idx=global_ep,
                     env_id=env_id,
                     N_GAMES=N_GAMES,
                     T_MAX=T_MAX,
                     dropout=dropout) for i in range(num_workers)]
    [w.start() for w in workers]
    [w.join() for w in workers]
