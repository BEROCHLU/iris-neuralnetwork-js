#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
import random
import datetime

DESIRED_ERROR = 0.001  # expected error
THRESHOLD = 10000  # epoch threshold
OUT_NODE = 1  # out node number
ETA = 0.5  # learning coefficient
ACTIVE = 0  # 0: sigmoid 1: ReLU


def sigmoid(a: float) -> float:
    if a < 0:
        return 1 - 1 / (1 + math.exp(a))
    else:
        return 1 / (1 + math.exp(-a))


dsigmoid = lambda a: a * (1 - a)
dmax = lambda a: a if (0 < a) == 1 else 0

IN_NODE, HID_NODE = None, None
hid, out = None, None
delta_hid, delta_out = None, None

epoch, days = 0, 0
fError = 0.05

x, t = None, None
v, w = [], []


def findHidOut(n: int):
    for i in range(HID_NODE):
        dot_h = 0
        for j in range(IN_NODE):
            dot_h += x[n][j] * v[i][j]
        if ACTIVE == 0:
            hid[i] = sigmoid(dot_h)
        elif ACTIVE == 1:
            hid[i] = max(0, dot_h)
        else:
            raise Exception("Activation function is 0 or 1")

    hid[HID_NODE - 1] = -1  # random.random() | -1

    for i in range(OUT_NODE):
        dot_o = 0
        for j in range(HID_NODE):
            dot_o += w[i][j] * hid[j]
        out[i] = sigmoid(dot_o)


def printResult():
    for i in range(days):
        findHidOut(i)
        rd_teacher = round(t[i][0], 3)
        rd_out = round(out[0], 3)
        print(f"teacher: {rd_teacher} out: {rd_out}")

    rd_err = round(fError, 5)
    f_time = time_ed - time_st

    if 60 <= f_time:
        n_minute = int(f_time / 60)
        f_sec = round(f_time % 60)
    else:
        n_minute = 0
        f_sec = round(f_time, 2)

    print(f"epoch: {epoch} final err: {rd_err}")
    print(f"time: {n_minute} min {f_sec} sec.")


def addBias(hsh: dict) -> dict:
    arrInput = hsh["input"]
    arrInput.append(random.random())  # add bias | random.random() * -1
    return arrInput


if __name__ == "__main__":
    json_path = "./json/xor.json"  # xor | cell30 | benchmark
    f = open(json_path, "r")
    arrHsh = json.load(f)

    x = list(map(addBias, arrHsh))
    t = list(map(lambda hsh: hsh["output"], arrHsh))

    IN_NODE = len(x[0])  # get input length include bias
    HID_NODE = IN_NODE + 1
    days = len(x)

    hid = [0] * HID_NODE
    out = [0] * OUT_NODE
    delta_hid = [0] * HID_NODE
    delta_out = [0] * OUT_NODE

    arrErr = []

    for i in range(HID_NODE):
        v.append([])
    for i in range(OUT_NODE):
        w.append([])

    for i in range(HID_NODE):
        for j in range(IN_NODE):
            v[i].append(random.random())  # random() | uniform(0.5, 1.0)
    for i in range(OUT_NODE):
        for j in range(HID_NODE):
            w[i].append(random.random())  # random() | uniform(0.5, 1.0)

    date_now = datetime.datetime.now()
    print(date_now.strftime("%F %T"))

    time_st = time.time()

    while DESIRED_ERROR < fError:
        epoch += 1
        fError = 0

        for n in range(days):
            findHidOut(n)

            for k in range(OUT_NODE):
                fError += 0.5 * (t[n][k] - out[k]) ** 2
                delta_out[k] = (t[n][k] - out[k]) * out[k] * (1 - out[k])

            for k in range(OUT_NODE):
                for j in range(HID_NODE):
                    w[k][j] += ETA * delta_out[k] * hid[j]

            for i in range(HID_NODE):
                delta_hid[i] = 0

                for k in range(OUT_NODE):
                    delta_hid[i] += delta_out[k] * w[k][i]

                if ACTIVE == 0:
                    delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i]
                elif ACTIVE == 1:
                    delta_hid[i] = dmax(hid[i]) * delta_hid[i]

            for i in range(HID_NODE):
                for j in range(IN_NODE):
                    v[i][j] += ETA * delta_hid[i] * x[n][j]
        # for in days
        if epoch % 100 == 0:
            print(f"epoch: {epoch} err: {round(fError, 5)}")

        if THRESHOLD <= epoch:
            print("force quit")
            break
    # while
    time_ed = time.time()
    printResult()
