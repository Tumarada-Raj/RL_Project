# Imports
import os
import torch
from ActorCritic import ActorCritic
from environment import KarelEnv
from utils import load_dataset, plot_lines
from policyNetwork import ActorCriticNet
from karel_agent import KarelAgent


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch_gen = torch.manual_seed(1998)

env_binary = KarelEnv()
env_complex = KarelEnv()

X_train, y_train = load_dataset(levels=["easy", "medium", "hard"], mode='train')
X_test, y_test = load_dataset(levels=["hard"], mode='val')
state_size = 16*11

train_config = dict(max_episodes=300000, early_stop=100, load_pretrained=False, verbose=True)

policy2 = ActorCriticNet(state_size, 6)
a2c_agent_complex = ActorCritic(policy2, env=env_complex, learn_by_demo=True, **train_config, variant_name='final_test1')
stats_complex = a2c_agent_complex.train(X_train, expert_traj=y_train, data_val=(X_test, y_test))

print("Base A2C Performance:")
a2c_agent_complex.evaluate(X_test, y_test, verbose=True)

print("Wrapped A2C Performance:")
karel_agent = KarelAgent(a2c_agent_complex, env_complex)
karel_agent.solve(X_test)
