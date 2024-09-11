
import os

import numpy as np

import gzip
import pandas as pd
import  math
import pandas as pa

cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))



def selected_tasks():
    cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    path = cur_path + '\\Machine_Workload\\task_days\\'

    # 读取数据
    df = pa.read_csv(path + "\\one_minute_3.csv", header=0, sep=',', encoding='utf-8', engine='python')
    tasks = df.values
    task_selected = []

    # list_IDs存储所有machine的ID
    # 300,000,000 是5分钟
    # 1000,000 是1s
    for i in range(len(tasks) - 1):
        if 864084799922 < tasks[i, 0] <950231884554:
            task_selected.append(tasks[i])

    dataframe = pa.DataFrame(data=list(task_selected), index=None, columns=None)
    dataframe.to_csv(cur_path + '\\Machine_Workload\\task_days\\' + 'selected_tasks.csv', index=False, sep=',')

    print('任务个数：', len(task_selected)) # 任务个数： 1038508
    print('分钟时间段：', (950231884554 - 864084799922) / 300000000) # 分钟时间段： 287.15694877333334

selected_tasks()