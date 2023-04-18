from common import Action
from typing import Union
from policyNetwork import ActorCriticNet
import torch
import numpy as np
import torch.optim as optim
import pickle


class ActorCritic:
    def __init__(
        self,
        policy: Union[ActorCriticNet, str],
        env,
        GAMMA=0.99,
        learning_rate=3e-4,
        alpha=0.5,
        clip_range=None,
        max_episodes=100000,
        max_eps_len=30,
        num_actions=6,
        learn_by_demo=True,
        early_stop=100,
        name='actor_critic',
        variant_name='v0',
        load_pretrained=False,
        verbose=False
    ):
        self.policy = policy
        self.env = env
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.clip_range = clip_range
        self.max_episodes = max_episodes
        self.max_eps_len = max_eps_len
        self.num_actions = num_actions
        self.learn_by_demo = learn_by_demo
        self.early_stop = early_stop
        self.name = name
        self.variant_name = variant_name
        self.verbose = verbose

        self.actor_optimizer = optim.Adam(self.policy.PI.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.policy.V.parameters(), lr=self.learning_rate)

        if load_pretrained:
            self.load_policy()

    def train(self, tasks, expert_traj=None, data_val=None):
        print("======================")
        print(f"Started training ")
        print("======================")
        all_lengths = []
        average_lengths = []
        best_accr = 0.0
        best_avg_extra_steps = 1e10
        worse_count = 0

        stats = {'loss': [], 'accr': []}

        for episode in range(self.max_episodes):

            task_state = tasks[episode % len(tasks)]
            if expert_traj:
                optimal_seq = expert_traj[episode % len(expert_traj)]["sequence"]
            else:
                optimal_seq = None
            task_state = self.env.reset(task_state)

            self.policy.train()
            t = self.generate_ep_rollout(task_state, optimal_seq)

            loss = self.update_agent()

            stats['loss'].append(loss)

            #all_rewards.append(np.sum(self.epidata["R"]))#
            all_lengths.append(t)
            average_lengths.append(np.mean(all_lengths[-10:]))
            if episode % 500 == 0:
                accr, avg_extra_steps = self.evaluate(data_val[0], data_val[1])
                stats['accr'].append(accr)
                if accr > best_accr or (accr == best_accr and avg_extra_steps < best_avg_extra_steps):
                    best_accr = accr
                    best_avg_extra_steps = avg_extra_steps
                    worse_count = 0
                    self.save_policy()
                    print("Policy updated!")
                else:
                    worse_count += 1

                if self.verbose:
                    print(f"Episode {episode}: validation accuracy={accr*100:.2f}%, \t avg_extra_steps={avg_extra_steps:.2f}")
                    print('-'*140)
                    if self.early_stop and worse_count >= self.early_stop:
                        print(
                            f"Early stopping triggered; validation accuracy hasn't increased for {worse_count} epsiodes!")
                        break

        print("========================")
        print(f"Finished training ")
        print("========================")
        self.load_policy()
        pickle.dump(stats, open(f'stats/stats_{self.variant_name}.pkl', 'wb'))
        return stats

    def save_policy(self):
        save_path = f"pretrained/{self.name}_{self.variant_name}.pth"
        torch.save(self.policy.state_dict(), save_path)
        #torch.save(self.policy, save_path)

    def load_policy(self):
        load_path = f"pretrained/{self.name}_{self.variant_name}.pth"
        self.policy.load_state_dict(torch.load(load_path))
        #torch.load(load_path)

    def evaluate(self, tasks, opt_seqs, H=50, verbose=False):
        solved = 0
        solved_opt = 0
        extra_steps = 0
        self.policy.eval()
        for i in range(len(tasks)):
            task_state = tasks[i]
            if opt_seqs:
                opt_seq = opt_seqs[i]["sequence"]

            state = self.env.reset(task_state)
            for t in range(H):
                action = self.act(state)
                state, reward, done, info = self.env.step(action)
                if done:
                    break

            solved += info['solved']
            solved_opt += (info['solved'] and t + 1 <= len(opt_seq))
            extra_steps += t - len(opt_seq) + 1 if reward > 0 else 0

        accr = solved / len(tasks)
        if opt_seqs:
            opt_accr = solved_opt/len(tasks)
            avg_extra_steps = extra_steps / len(tasks)
            opt_stats_str = f", Accuracy(optimally solved)={opt_accr*100:.2f}%, avg extra steps={avg_extra_steps:.2f}"
        else:
            opt_stats_str = ""

        print(f"Attempted {len(tasks)} tasks, correctly solved {solved}. Accuracy(solved)={accr*100:.2f}%{opt_stats_str}")
        return accr, avg_extra_steps

    def reset_rollout_buffer(self):
        self.epidata = {"S": [], "V": [], "A": [], "R": [],
                        "S_next": [], "G": [], "PI": [], "D": []}

    def update_rollout_buffer(self, state, state_value, action, reward, state_next, act_prob, done):
        self.epidata["S"].append(state)
        self.epidata["V"].append(state_value)
        self.epidata["A"].append(action)
        self.epidata["R"].append(reward)
        self.epidata["S_next"].append(state_next)
        self.epidata["PI"].append(act_prob)
        self.epidata["D"].append(done)

    def wrap_up_episode(self):
        self.epidata["S"] = torch.from_numpy(np.array(self.epidata["S"]))
        self.epidata["S_next"] = torch.from_numpy(
            np.array(self.epidata["S_next"]))

        self.epidata["G"] = self.compute_returns(
            self.epidata["R"], self.epidata["S"])
        self.epidata["R"] = torch.FloatTensor(self.epidata["R"])

        if None not in self.epidata["V"]:
            self.epidata["V"] = torch.cat(self.epidata["V"])
        self.epidata["D"] = torch.IntTensor(self.epidata["D"])

    def generate_ep_rollout(self, state, optimal_seq=None):
        self.reset_rollout_buffer()
        for t in range(self.max_eps_len):
            action_dist, state_value = self.policy(state)
            #dist = action_dist.cpu().detach().numpy()

            if self.learn_by_demo and optimal_seq:
                # Use expert optimal action
                action = Action.from_str[optimal_seq[t]]
            else:
                # Use own agent's action
                dist = action_dist.cpu().detach().numpy()
                action = np.random.choice(self.num_actions, p=np.squeeze(dist))

            act_prob = action_dist[0][action]
            new_state, reward, done, _ = self.env.step(action)

            self.update_rollout_buffer(
                state, state_value, action, reward, new_state, act_prob, done)
            state = new_state

            if done:
                break
        self.wrap_up_episode()
        return t

    def compute_returns(self, rewards, states):
        G = 0.0
        Gs = np.zeros_like(rewards, dtype=float)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.GAMMA * G
            Gs[t] = G
        return torch.FloatTensor(Gs)


    def compute_actor_loss(self, data):
        probs, G, v = data["PI"], data["G"], data["V"]
        probs = torch.stack(probs)
        log_pi = -torch.log(probs + 1e-13)

        if self.learn_by_demo:
            adv = 1.0
        else:
            adv = G - v.detach()

        loss_pi = self.alpha * (log_pi * adv).mean()

        return loss_pi

    def compute_critic_loss(self, data):
        G, v = data['G'], data['V']

        alpha = 0.5
        # MSE loss against Bellman backup
        loss_q = alpha * ((G - v).pow(2)).mean()

        return loss_q

    def update_agent(self):
        actor_loss = self.compute_actor_loss(self.epidata)
        critic_loss = self.compute_critic_loss(self.epidata)
        # ac_loss = actor_loss + critic_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item()

    def act(self, state):
        action_dist, _ = self.policy(state)
        dist = action_dist.detach().numpy()
        action = dist.argmax()
        return action

    def judge(self, env, action=None):
        next_state, reward, done, _ = env.probe(action)
        next_state_value = self.policy.V(next_state).item() if not done else 0
        return reward + next_state_value
