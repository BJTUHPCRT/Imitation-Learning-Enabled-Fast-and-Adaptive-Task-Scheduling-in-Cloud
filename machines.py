
import pandas as pa
import os
import pandas
import numpy

# 只有一个 bool 值等于1

print(os.getcwd())
cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def real_machine_events():
    machine_a = pa.read_csv(cur_path + '\\Machine_Workload\\machines\\machineAttributes.csv', sep=',',encoding='utf-8',engine='python')
    machine_a_event = machine_a.values
    list_2 = []
    list_3_0 = []
    list_3_1 = []
    list_3_str = []
    list_3_digital = []
    list_4 = []
    MachineID = []
    # 把attribute delete的删除掉
    for i in range(len(machine_a_event)):
        # if machine_a_event[i, 2] not in list_2:
        #     list_2.append(machine_a_event[i, 2])
        # print(machine_a_event[i, 3])
        if  machine_a_event[i, 4] == 0:
            if machine_a_event[i, 3].isdigit():
                list_3_digital.append(machine_a_event[i, 3])
            else:
                list_3_str.append(machine_a_event[i, 3])
                MachineID.append(machine_a_event[i])
        if  machine_a_event[i, 4] == 1:
            if machine_a_event[i, 3] not in list_3_1:
                list_3_1.append(machine_a_event[i, 3])
        # if machine_a_event[i, 4] not in list_4:
        #     list_4.append(machine_a_event[i, 4])

    # print('list 2', len(list_2))
    print('list 3 digital', len(list_3_digital))
    print('list 3 string', len(list_3_str))
    # print('list 4', len(list_4))
    dataframe = pandas.DataFrame(data=list(MachineID), index=None, columns=None)
    dataframe.to_csv(cur_path + '\\Machine_Workload\\machines\\machineID_Attributes.csv', index=False, sep=',')

#  只有3列是字符串的
def machine_0():
    MachineID_single = []
    machines_ID = pa.read_csv(cur_path + '\\Machine_Workload\\machines\\machineID_Attributes.csv', sep=',', encoding='utf-8',
                            engine='python')
    machine_IDs = machines_ID.values
    machine_single = []
    print('events number:', len(machine_IDs))

    # # 已经删除了delete的machine
    # for i in range(len(machine_IDs)):
    #     if machine_IDs[i, 3] not in machine_single:
    #         machine_single.append(machine_IDs[i, 3])
    #
    # print(len(machine_single))
    # dataframe = pandas.DataFrame(data=list(machine_single), index=None, columns=None)
    # dataframe.to_csv(cur_path + '\\Machine_Workload\\machines\\machineID_Single.csv', index=False, sep=',')

# GK约束 + 3列
def machine_1():
    MachineID_single = []
    machines_GK = pa.read_csv(cur_path + '\\Machine_Workload\\machines\\machineID_GK.csv', sep=',', encoding='utf-8',
                            engine='python')
    machine_GKs = machines_GK.values
    machine_single = []
    print('events number:', len(machine_GKs))

    # 已经删除了delete的machine
    for i in range(len(machine_GKs)):
        from string import Template
        s = Template('${s1}${s2}')
        tmp = s.safe_substitute(s1=machine_GKs[i, 2], s2=machine_GKs[i, 3])
        print(len(machine_single))
        if tmp not in MachineID_single:
            MachineID_single.append(tmp)
            machine_single.append(machine_GKs[i])

    print(len(machine_single))
    dataframe = pandas.DataFrame(data=list(machine_single), index=None, columns=None)
    dataframe.to_csv(cur_path + '\\Machine_Workload\\machines\\machineID_Single.csv', index=False, sep=',')

def machine_GKrecords():
    MachineID_single = []
    machines_ID = pa.read_csv(cur_path + '\\Machine_Workload\\machines\\machineAttributes.csv', sep=',', encoding='utf-8',
                            engine='python')
    machine_IDs = machines_ID.values
    machine_single = []
    # print('events number:', len(machine_IDs))

    # 已经删除了delete的machine
    for i in range(len(machine_IDs)):
        # if
        if machine_IDs[i, 2][:2]  == 'GK':
            machine_single.append(machine_IDs[i])

    print(len(machine_single))
    dataframe = pandas.DataFrame(data=list(machine_single), index=None, columns=None)
    dataframe.to_csv(cur_path + '\\Machine_Workload\\machines\\machineID_GK.csv', index=False, sep=',')

def machine_events_table():
    machine_a = pa.read_csv(cur_path + '\\Machine_Workload\\machines\\machineEvents.csv', sep=',', encoding='utf-8',
                            engine='python')
    machine_a_event = machine_a.values
    list_IDs = []
    machine_IDS = []
    timesteps = []

    # 获得时间戳 + machineID
    for i in range(len(machine_a_event)):
        #取出timesteps
        if machine_a_event[i, 0] not in timesteps:
            timesteps.append(machine_a_event[i, 0])

        if machine_a_event[i, 1] not in list_IDs:
            list_IDs.append(machine_a_event[i, 1])
            machine_IDS.append(machine_a_event[i])
        # print(machine_a_event[i, 3])

    counter = 0
    for i in range(len(list_IDs)):
        tmp_machine = []
        for j in range(len(machine_a_event)):
            if machine_a_event[j, 1] ==  list_IDs[i]:
                tmp_machine.append(machine_a_event[j])


        # dataframe = pandas.DataFrame(data=list(tmp_machine), index=None, columns=None)
        # dataframe.to_csv(cur_path + '\\Machine_Workload\\machinesEvent\\' + str(list_IDs[i]) + 'machineID_events.csv', index=False, sep=',')

    print('event machineID', len(list_IDs))

    # dataframe = pandas.DataFrame(data=list(machine_IDS), index=None, columns=None)
    # dataframe.to_csv(cur_path + '\\Machine_Workload\\machinesEvent\\Atimesteps.csv', index=False, sep=',')

    # dataframe = pandas.DataFrame(data=list(machine_IDS), index=None, columns=None)
    # dataframe.to_csv(cur_path + '\\Machine_Workload\\machines\\machineIDs_events.csv', index=False, sep=',')

def statics():

    machine_a = pa.read_csv(cur_path + '\\Machine_Workload\\machines\\machineEvents.csv', sep=',', encoding='utf-8',
                            engine='python')
    machine_a_event = machine_a.values
    list_IDs = []
    machine_IDS = []
    list_timesteps = []

    # list_IDs存储所有machine的ID
    for i in range(len(machine_a_event)):
        if machine_a_event[i, 1] not in list_IDs:
            list_IDs.append(machine_a_event[i, 1])
            machine_IDS.append(machine_a_event[i])

    # # 获得时间戳 + machineID
    # for i in range(len(machine_a_event)):
    #     # 取出timesteps
    #     if machine_a_event[i, 0]  == 0:
    #         machine_IDS.append(machine_a_event[i])
    #         # 时间戳为0的machineID
    #         if machine_a_event[i, 1] not in list_IDs:
    #             list_IDs.append(machine_a_event[i, 1])
    #
    # print('时间戳为0的events数量：', len(machine_IDS))
    # print('machine number', len(list_IDs))
    # #若二者相等，则所有机器都打开

    counter_0 = 0
    counter_1 = 0
    counter = 0
    #每个machine的轨迹
    for i in range(len(list_IDs)):
        tmp_machine = []
        for j in range(len(machine_a_event)):
            if machine_a_event[j, 1] ==  list_IDs[i]:
                tmp_machine.append(machine_a_event[j, 2])
        # tmp_machine 中存储了machine的轨迹
        if 0 in tmp_machine:
            counter += 1
        if 1 in tmp_machine:
            counter_0 += 1
        if 2 in tmp_machine:
            counter_1 += 1
    #start 的时候有12476个machine被启动
    print('有过增加操作的：', counter)  # 12583   100%
    print('有过移除操作的：', counter_0)  # 5141  40.9%
    print('有过更新操作的：', counter_1)   # 1268   10.1%

def oneDay_machine():
    # machine_a = pa.read_csv(cur_path + '\\Machine_Workload\\machineDay', sep=',', encoding='utf-8',
    #                         engine='python')
    # machine_a_event = machine_a.values
    # list_IDs = []
    # machine_IDS = []

    # list_IDs存储所有machine的ID
    # for i in range(len(machine_a_event)):
    #     if machine_a_event[i, 1] not in list_IDs:
    #         list_IDs.append(machine_a_event[i, 1])
    #         # machine_IDS.append(machine_a_event[i])
    # print('涉及到的machine数量', len(list_IDs))

    cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    path = cur_path + '\\Machine_Workload\\machineDay\\'

    # 读取数据
    files = os.listdir(path)
    # for file in files:
    # if os.path.exists(path + "\\" + file) and os.path.isdir(path + "\\" + file):
    tempPath = path
    subFiles = os.listdir(tempPath)
    # subFiles.sort(key=lambda x: int(x[:-4]))
    total_cpu = []
    for subFile in subFiles:
        df = pa.read_csv(tempPath + "\\" + subFile, header=0, sep=',', encoding='utf-8', engine='python')
        machines = df.values
        list_IDs = []
        list_timesteps = []

        # list_IDs存储所有machine的ID
        for i in range(len(machines)):
            if machines[i, 1] not in list_IDs:
                list_IDs.append(machines[i, 1])
                # machine_IDS.append(machine_a_event[i])
        # print('一天内涉及到的machine数量', len(list_IDs))
        # print('百分比：', len(list_IDs) / 12583 )


        counter_0 = 0
        counter_1 = 0
        counter = 0
        # 每个machine的轨迹
        for i in range(len(list_IDs)):
            tmp_machine = []
            for j in range(len(machines)):
                if machines[j, 1] == list_IDs[i]:
                    tmp_machine.append(machines[j, 2])
            # tmp_machine 中存储了machine的轨迹
            if 0 in tmp_machine:
                counter += 1
            if 1 in tmp_machine:
                counter_0 += 1
            if 2 in tmp_machine:
                counter_1 += 1
        # start 的时候有12476个machine被启动
        print('一天内有几次增加操作：', counter)  # 12583   100%
        print('一天内有几次移除操作：', counter_0)  # 5141  40.9%
        print('一天内有几次更新操作：', counter_1)  # 1268   10.1%
        print('-----------------------------------------------------')

# 计算machine event之间的间隔时间
def computeInternal():
    cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    path = cur_path + '\\Machine_Workload\\machineDay\\'

    # 读取数据
    df = pa.read_csv(path + "\\10.csv", header=0, sep=',', encoding='utf-8', engine='python')
    machines = df.values
    machine_internals = []

    # list_IDs存储所有machine的ID
    # 300,000,000 是5分钟
    # 1000,000 是1s
    for i in range(len(machines) - 1):
        internal = (machines[i + 1, 0] - machines[i, 0]) / 1000000
        if machines[i, 2] == 2:
            from random import choice
            a = [0, 1]
            machine_event = choice(a)
        else:
            machine_event = machines[i, 2]
        machine_internal = []
        # 将时间戳放进去方便找时间段
        machine_internal.append(machines[i, 0])
        # 存储有操作的machine的ID
        ID = numpy.random.randint(low=0, high=1000, size=1)
        machine_internal.append(ID[0]) # down机从这里开始，start也是从这里开始
        machine_internal.append(internal)
        machine_internal.append(machine_event)
        machine_internals.append(machine_internal)

    dataframe = pandas.DataFrame(data=list(machine_internals), index=None, columns=None)
    dataframe.to_csv(cur_path + '\\Machine_Workload\\machines\\' +  'machine_timesteps_events.csv', index=False, sep=',')

# 计算task到达的间隔时间
def computeTaskInternal():
    cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    path = cur_path + '\\Machine_Workload\\task_days\\'

    # 读取数据
    df = pa.read_csv(path + "\\one_minute_2.csv", header=0, sep=',', encoding='utf-8', engine='python')
    tasks = df.values
    task_internals = []

    # list_IDs存储所有machine的ID
    # 300,000,000 是5分钟
    # 1000,000 是1s
    for i in range(len(tasks) - 1):
        internal = (tasks[i + 1, 1] - tasks[i, 1]) / 1000000
        # task_event = tasks[i, 2]

        task_internal = []
        # 将时间戳放进去方便找时间段
        task_internal.append(tasks[i, 1])
        task_internal.append(internal)
        # task_internal.append(task_event)
        task_internals.append(task_internal)

    # dataframe = pandas.DataFrame(data=list(task_internals), index=None, columns=None)
    # dataframe.to_csv(cur_path + '\\Machine_Workload\\task_days\\' +  'task_timesteps_events.csv', index=False, sep=',')
# machine_events_table()
# statics()
oneDay_machine()
# computeInternal()
# computeTaskInternal()
