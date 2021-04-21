import numpy as np
import os
from datetime import datetime


def create_folder(path):
    if not os.path.exists(path=path):
        os.mkdir(path=path)


def create_results_folder():
    path = f"results"
    create_folder(path=path)
    date = datetime.today().strftime('%d-%m-%y')
    path = os.path.join(path, date)
    create_folder(path=path)
    return path


def save_data(x, name, folder="results"):
    x_numpy = x.detach().cpu().numpy()
    np.save(f"{folder}/%s.npy" % name, x_numpy)
    pass
