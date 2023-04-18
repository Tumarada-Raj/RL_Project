#%%
import os
import json
import numpy as np
from common import Action, Direction
from sklearn.utils import shuffle
import pickle as pkl
import plotly.graph_objects as go

data = {"easy": 'data_easy', "medium": 'data_medium', "hard": 'data'}


def load_dataset(levels=["easy"], mode="train"):  #, compact=True):
    #size_mode = 'compact' if compact else 'verbose'
    size_mode = 'verbose'
    pickled_path = f'datasets/preprocessed_{mode}_{"_".join(levels)}_{size_mode}.pkl'
    if os.path.isfile(pickled_path):
        all_tasks, all_seqs = pkl.load(open(pickled_path, 'rb'))
        return all_tasks, all_seqs
    
    all_tasks, all_seqs = [], []
    for level in levels:
        data_dir = f"datasets/{data[level]}"
        tasks_dir = f"{data_dir}/{mode}/task"
        seqs_dir = f"{data_dir}/{mode}/seq"
        task_fnames = sorted(os.listdir(tasks_dir), key=lambda s: int(s.split("_")[0]))
        seq_fnames = sorted(os.listdir(seqs_dir), key=lambda s: int(s.split("_")[0]))

        tasks = [json.load(open(f"{tasks_dir}/{fname}")) for fname in task_fnames]
        seqs = [json.load(open(f"{seqs_dir}/{fname}")) for fname in seq_fnames]

        all_tasks.extend(tasks)
        all_seqs.extend(seqs)

    all_tasks, all_seqs = shuffle(all_tasks, all_seqs, random_state=73)

    if pickled_path:
        pkl.dump((all_tasks, all_seqs), open(pickled_path, 'wb'))
    return all_tasks, all_seqs


def load_dir(tasks_dir):
    all_tasks = []
    task_fnames = sorted(os.listdir(tasks_dir), key=lambda s: int(s.split("_")[0]))
    task_ids = list(map(lambda s: s.split("_")[0], task_fnames))
    tasks = [json.load(open(f"{tasks_dir}/{fname}")) for fname in task_fnames]
    all_tasks.extend(tasks)
    return all_tasks, task_ids


def stateSpaceVector(task):   #, is_compact=True):
    r, c = task["gridsz_num_rows"], task["gridsz_num_cols"]
    pregrid_agent_r_loc = task["pregrid_agent_row"]
    pregrid_agent_c_loc = task["pregrid_agent_col"]
    pregrid_agent_dir = Direction.from_str[task["pregrid_agent_dir"]]
    postgrid_agent_r_loc = task["postgrid_agent_row"]
    postgrid_agent_c_loc = task["postgrid_agent_col"]
    postgrid_agent_dir = Direction.from_str[task["postgrid_agent_dir"]]

    features = np.zeros((r, c, 1 + 1 * 2 + 2 * 4))

    features[pregrid_agent_r_loc, pregrid_agent_c_loc, pregrid_agent_dir] = 1
    features[postgrid_agent_r_loc, postgrid_agent_c_loc, postgrid_agent_dir + 4] = 1

    # Represent walls
    for wall in task["walls"]:
        features[wall[0], wall[1], 8] = 1

    # Represent markers
    for marker in task["pregrid_markers"]:
        features[marker[0], marker[1], 9] = 1

    for marker_post in task["postgrid_markers"]:
        features[marker_post[0], marker_post[1], 10] = 1

    return features


def sequenceVectors(seq):
    feat_v = [Action.from_str[cmd] for cmd in seq["sequence"]]
    return feat_v


def plot_lines(groups, axes_titles=['x', 'y'], title='Figure'):
    fig = go.Figure()
    for name in groups:
        fig.add_trace(go.Scatter(x=groups[name][0], y=groups[name][1],
                                 mode='lines',
                                 name=name))

    fig.update_layout(title=title,
                      xaxis_title=axes_titles[0],
                      yaxis_title=axes_titles[1])
    fig.show()

