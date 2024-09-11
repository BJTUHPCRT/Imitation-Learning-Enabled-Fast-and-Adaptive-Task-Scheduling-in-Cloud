
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
