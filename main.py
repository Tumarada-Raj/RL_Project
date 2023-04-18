import torch
from ActorCritic import ActorCritic
from environment import KarelEnv
from karel_agent import KarelAgent
from utils import load_dir
from policyNetwork import ActorCriticNet
from argparse import ArgumentParser
import json
import pickle as pkl
import os
from sklearn.utils import shuffle

torch_gen = torch.manual_seed(1998)

parser = ArgumentParser()
parser.add_argument("-o", "--output-dir-path", dest="output_dir",
                    help="Path to save sequences of unseen tasks", default="C:/Users/rajmo/Desktop/RL/Project/Tumarada_project2_train/datasets/data/test_without_seq/generated_seq")
parser.add_argument("-l", "--dataset-level", dest="level",
                    help="Dataset level (easy, medium, hard)", default="hard")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--task-path", dest="task_path",
                    help="Path to Karel task", metavar="C:/Users/rajmo/Desktop/RL/Project/Tumarada_project2_train/datasets/data/test_without_seq/task/200000_task.json")
group.add_argument("-d", "--dataset-path", dest="dataset_dir",
                    help="Path to Karel dataset folder", metavar="C:/Users/rajmo/Desktop/RL/Project/Tumarada_project2_train/datasets/data/test_without_seq/task")


args = parser.parse_args()

data = {"easy": 'data_easy', "medium": 'data_medium', "hard": 'data'}


def load_dataset_test(levels=["easy"], mode="train"):  # , compact=True):
    # size_mode = 'compact' if compact else 'verbose'
    size_mode = 'verbose'
    pickled_path = f'datasets/preprocessed_{mode}_{"_".join(levels)}_{size_mode}.pkl'
    if os.path.isfile(pickled_path):
        all_tasks, all_seqs = pkl.load(open(pickled_path, 'rb'))
        return all_tasks, all_seqs

    all_tasks, all_ids = [], []
    for level in levels:
        data_dir = f"datasets/{data[level]}"
        tasks_dir = f"{data_dir}/{mode}/task"
        seqs_dir = f"{data_dir}/{mode}/seq"
        task_fnames = sorted(os.listdir(tasks_dir), key=lambda s: int(s.split("_")[0]))
        ids = list(map(lambda s: s.split("_")[0], task_fnames))

        tasks = [json.load(open(f"{tasks_dir}/{fname}")) for fname in task_fnames]
        #seqs = [json.load(open(f"{seqs_dir}/{fname}")) for fname in seq_fnames]

        all_tasks.extend(tasks)
        all_ids.extend(ids)

    all_tasks, all_seqs = shuffle(all_tasks, all_ids, random_state=73)

    if pickled_path:
        pkl.dump((all_tasks, all_ids), open(pickled_path, 'wb'))
    return all_tasks, all_ids


if args.task_path:
    X_test = [json.load(open(args.task_path))]
    task_ids = None
    output_dir = None
else:
    X_test, task_ids = load_dataset_test(levels=[args.level], mode='test_without_seq')
    #print("Task Ids:", task_ids)
    #X_test, task_ids = load_dataset1(levels=["hard"], mode='val')
    #X_test, task_ids = load_dir(args.dataset_dir)
    output_dir = args.output_dir

state_size = 16*11
env_complex = KarelEnv()

config = dict(load_pretrained=True, verbose=True)

a2c_policy = ActorCriticNet(state_size, 6)
a2c_agent = ActorCritic(a2c_policy, env=env_complex, **config, variant_name='final_test1')


karel_agent = KarelAgent(a2c_agent, env_complex, cycle_detection_enabled=True, env_probe_enabled=True)
karel_agent.solve(X_test, output_dir=output_dir, task_ids=task_ids)
#karel_agent.solve(X_test)