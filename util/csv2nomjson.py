# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import json
import os
import pandas as pd

R_TRAIN_PATH = "../csv/irisnom-train.csv"
R_TEST_PATH = "../csv/irisnom-test.csv"
W_TRAIN_PATH = "../json/iris-train.json"
W_TEST_PATH = "../json/iris-test.json"

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 実行ファイルフォルダをカレントフォルダに変更

for tpl in [(R_TRAIN_PATH, W_TRAIN_PATH), (R_TEST_PATH, W_TEST_PATH)]:
    lst_dc = []
    df_iris = pd.read_csv(tpl[0], names=["x0", "x1", "x2", "x3", "t0", "t1", "t2"])

    lst_x0 = df_iris["x0"].values.tolist()
    lst_x1 = df_iris["x1"].values.tolist()
    lst_x2 = df_iris["x2"].values.tolist()
    lst_x3 = df_iris["x3"].values.tolist()
    lst_t0 = df_iris["t0"].values.tolist()
    lst_t1 = df_iris["t1"].values.tolist()
    lst_t2 = df_iris["t2"].values.tolist()

    for x0, x1, x2, x3, t0, t1, t2 in zip(lst_x0, lst_x1, lst_x2, lst_x3, lst_t0, lst_t1, lst_t2):
        dc = {"input": [x0, x1, x2, x3], "output": [t0, t1, t2]}
        lst_dc.append(dc)

    with open(tpl[1], "w") as f:
        json.dump(lst_dc, f, indent=4)

print("Done csv to json")
