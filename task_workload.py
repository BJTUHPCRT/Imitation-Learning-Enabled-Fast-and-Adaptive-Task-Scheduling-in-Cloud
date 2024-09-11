"""
Loading Google cluster-2011 dataset for experiment
"""
from __future__ import print_function

import os

import numpy as np

import gzip
import pandas as pd
import  math

cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def save_machineDay(machineDay, day):
    import sys
    import os
    import pandas
    path  = cur_path + '\\Machine_Workload\\task_days\\'
    dataframe = pandas.DataFrame(data=list(machineDay), index=None, columns=None)
    x = dataframe.to_csv(path + str(day) + '.csv', sep=',',index=None)

def load_eventsTable(path):

    files = os.listdir(path)
    job_events = []
    machine_attributes = []
    machine_events = []
    task_constraints = []
    task_events = []
    cpu_workload = []
    task_usage = []
    counter = 0
    week = 0
    all_task = []

    for file in files:
        if file == 'task_events':
            week = 1
            #if os.path.exists(path + "\\" + file) and os.path.isdir(path + "\\" + file):
            tempPath = path + "/" + file
            subFiles = sorted(os.listdir(tempPath))
            # subFiles.sort(key=lambda x:int(x[:-4]))
            total_cpu = []
            for subFile in subFiles:
                #if os.path.exists(tempPath + "\\" + subFile):
                # df = pd.read_csv(tempPath + "/" + subFile, compression='gzip', header=0, sep=',', quotechar='" ',error_bad_lines=False)
                # df = pd.read_csv(tempPath + "/" + subFile, header=0, sep=',', encoding='utf-8', engine='python')
                csv_path = tempPath + "/" + subFile
                df = pd.read_csv(csv_path, compression='gzip')
                # with gzip.open(tempPath + "/" + subFile, 'rb') as f:
                #     pd.read_csv('data.csv.gz', compression='gzip')
                #     tempdata = f.read()
                #     df = np.frombuffer(f.read(), np.uint8, offset=16)
                for i in range(len(df)):
                    all_task.append(df.values[i])
                # compute_cpu_workload(task_event,str(counter),week)
                # counter += 1

    # dataframe = pd.DataFrame(data=list(all_task), index=None, columns=None)
    # x = dataframe.to_csv(cur_path + '\\Machine_Workload\\task_days\\AllTask.csv', sep=',', index=None)
    #获取到所有rask，接下来进行按天切分
    compute_cpu_workload()

def compute_cpu_workload(task_event,day,week):
    week = 0
    interval = 300000000 * 288 #间隔5min 300000000
    workload_list = []
    workload_list_tmp = []
    task_list_tmp = []
    change_task = []
    cpu_list = []
    tmp_counter = 0
    task_counter_list = []
    done = False
    task_number = 0
    counter = 0
    current_task = None
    task_list_end = task_event[len(task_event) - 1, 0]
    k = 0

    start = task_event[0, 0]
    end = start + interval

    co = int(task_list_end/interval) + 1

    # 所有的数据
    while True:
        total_internal_workload = 0
        task_list_tmp = []
        task_counter_minute = 0
        day = 0

        #计算间隔时间内的负载总和
        while task_event[tmp_counter, 0] < end:
            #最后一段的情况
            if end > task_list_end:
                while task_event[tmp_counter, 0] < task_list_end:
                    # total_internal_workload += task_event[tmp_counter, 9]
                    # cpu_list.append(task_event[tmp_counter, 9])
                    # tmp_counter += 1
                    # task_counter_minute += 1 #记录每分钟有多少task

                    # 这个workload对应的一波task
                    task_list_tmp.append(task_event[tmp_counter])

                done = True
                # printWeekDataset_tmp(workload_list_tmp, str(tmp_counter), day)
                break
            else:
                # if not math.isnan(task_event[k].astype(np.int)):
                # total_internal_workload += task_event[tmp_counter, 9]
                # cpu_list.append(task_event[tmp_counter, 9])
                # tmp_counter += 1
                # task_counter_minute += 1

                #这个workload对应的一波task
                task_list_tmp.append(task_event[tmp_counter])
            # if 300 <= task_number <= 800:
            #     change_task.append(task_event[tmp_counter])

        save_machineDay(task_list_tmp, day)
        day += 1

        start = end
        end = start + interval
        #结尾跳出
        if done: break

def day_tasks(path):

    files = os.listdir(path)
    job_events = []
    machine_attributes = []
    machine_events = []
    task_constraints = []
    task_events = []
    cpu_workload = []
    task_list_tmp = []
    counter = 0
    day = 0
    all_task = []

    interval = 300000000 * 288  # 间隔5min 300000000
    start = 0
    end = start + interval

    for file in files:
        if file == 'task_events':
            week = 1
            #if os.path.exists(path + "\\" + file) and os.path.isdir(path + "\\" + file):
            tempPath = path + "/" + file
            subFiles = sorted(os.listdir(tempPath))
            # subFiles.sort(key=lambda x:int(x[:-4]))
            total_cpu = []
            for subFile in subFiles:
                #if os.path.exists(tempPath + "\\" + subFile):
                # df = pd.read_csv(tempPath + "/" + subFile, compression='gzip', header=0, sep=',', quotechar='" ',error_bad_lines=False)
                # df = pd.read_csv(tempPath + "/" + subFile, header=0, sep=',', encoding='utf-8', engine='python')
                csv_path = tempPath + "/" + subFile
                df = pd.read_csv(csv_path, compression='gzip')

                for i in range(len(df)):
                    if df.values[i, 0] <= end:
                        # df.to_csv(cur_path + '\\Machine_Workload\\task_days\\' + str(day) + '.csv', mode='a',
                        #                  header=False)
                        # print(df.values[i, 9])
                        if math.isnan(df.values[i, 9]):
                            continue
                        else:
                            data = []
                            data.append(df.values[i, 0])
                            data.append(df.values[i, 9])
                            data.append(df.values[i, 10])
                            # dataframe = pd.DataFrame(columns=['timestamp', 'cpu','memory'], data=data)
                            # dataframe = pd.DataFrame(data=np.array(df.values[i]).reshape(-1,len(df.values[i])), index=None, columns=None)
                            # dataframe = pd.DataFrame(data=df.values[i], index=None, columns=None).T
                            dataframe = pd.DataFrame(data=data, index=None, columns=None).T
                            dataframe.to_csv( cur_path + '\\Machine_Workload\\task_days\\' + str(day) + '.csv', mode='a', header=False)
                    else:
                        # save_machineDay(task_list_tmp, day)
                        # task_list_tmp = []
                        day += 1
                        start = end
                        end = start + interval

                # for i in range(len(df)):
                #     if df.values[i, 0] <= end:
                #         # data = df.values[i, 0]
                #         dataframe = pd.DataFrame(data=df.iloc[i], index=None, columns=None)
                #         dataframe.to_csv( cur_path + '\\Machine_Workload\\task_days\\' + str(day) + '.csv', mode='a', header=False)
                #         # with open( cur_path + '\\Machine_Workload\\task_days\\' + str(day) + '.csv', mode='a', header=False) as f:
                #             # for line in s:
                #         # f.write(str(df.values[i]))
                #         # task_list_tmp.append(df.values[i])
                #     else:
                #         # save_machineDay(task_list_tmp, day)
                #         # task_list_tmp = []
                #         day += 1
                #         start = end
                #         end = start + interval


path = "E:/experiment/clusterdata-2011-2/"
day_tasks(path)
# load_eventsTable(path)
# task_events, machine_events = load_eventsTable(path)
# print(machine_events,task_events)