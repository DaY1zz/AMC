import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import bisect
import random
import time
import sys
import polars as pl
from simple_pid import PID
import pickle as pkl
from multiprocessing import Pool, cpu_count

sys.path.append('/home/dell/xyf/azurefunctions-dataset2019/analysis')
sys.path.append('/home/dell/xyf/AMC')
from common import *

if __name__ == "__main__": 
    with open('/home/dell/xyf/AMC/variables_checkpoint.pkl', "rb") as file:
        train_func_ids, test_func_arrcount, func_class, func_lst, func_corr_lst, test_func_corr, test_func_corr_perform= pkl.load(file)

    # 筛选出可预测函数并根据训练集进行初步预测
    PE_THRESHOLD = 0.2

    CV_WT_UPPER_THRESHOLD = 5
    CV_WT_LOWER_THRESHOLD = 2

    LOCAL_WINDOW = 60*48
    PREDICT_WINDOW = 60

    PID_TARGET = 0.15   #冷启动率
    T_ALPHA = 0.2       #浮动参数
    BETA = 0.1          #实例参数

    HISTORY_TIMEOUT = 12*60     # 12小时  
    HISTORY_LENGTH = 6          # 6条调用记录
    IAT_MIN = 1
    IAT_QUANTILE = 0.8     #IAT置信分位数

    df = pl.read_csv("/home/dell/xyf/AMC/func_info.csv")
    df = df.filter(pl.col('Type') != 2)   #过滤regular

    pe_df = df.filter(pl.col('PE') > PE_THRESHOLD)  #   PE > 0.2
    cv_WT_df = df.filter((pl.col('CV_WT') > CV_WT_LOWER_THRESHOLD))\
            .filter((pl.col('CV_WT') < CV_WT_UPPER_THRESHOLD))\
            .filter(~pl.col('CV_WT').is_nan())\
            .filter(pl.col('PE')> 0.1)\
                                                    #  2 < CV_WT < 5 && PE > 0.1
    df_union = pl.concat([pe_df, cv_WT_df]).unique()

    predictable_func_ids = df_union.select('Function').to_numpy().flatten().tolist()
    predictable_func_ids = set(predictable_func_ids)
    pred_func_account = {}
    # boosting_params = {
    #                         "objective": "regression",
    #                         "metric": "mape",
    #                         "verbosity": -1,
    #                         "boosting_type": "gbdt",
    #                         "seed": 42,
    #                         "learning_rate": 0.1,
    #                         "min_child_samples": 4,
    #                         "num_leaves": 128,
    #                         "num_iterations": 100
    #                                     }
    # with tqdm(total=len(predictable_func_ids)) as pbar:
    #     for func in predictable_func_ids:
    #         lp_model = lp.LazyProphet(scale=True,
    #                         seasonal_period=[24, 168],
    #                         n_basis=8,
    #                         fourier_order=10,
    #                         ar=list(range(1, 97)),
    #                         decay=.99,
    #                         linear_trend=None,
    #                         decay_average=False,
    #                         boosting_params=boosting_params
    #                         )
            
    #         arr = train_func_arrcount[func]
    #         window_data = arr[-LOCAL_WINDOW:]
    #         lp_model.fit(window_data)            
    #         pred_result = lp_model.predict(PREDICT_WINDOW).flatten()
    #         pred_result = list(map(lambda x: round(x) if x > 0 else 0, pred_result))
    #         pred_func_account[func] = pred_result
    #         pbar.update(1)
    with open('/home/dell/xyf/AMC/method_exp/prediction_result/LazyProhet/0_59_result.pkl', 'rb') as f:
        pred_func_account = pkl.load(f)

    memory = set()

    for func in train_func_ids: # Pre_warm for testing at time 0
        _type = func_class[func].type

        if _type == REGULAR and func_class[func].last_call == -1 \
            and (func_class[func].lasting_info["max"] > 1 or func_class[func].lasting_info["seq_num"] == 1):
            memory.add(func)
            func_class[func].load(0, 1)
        if _type == REGULAR:
            for (left, right) in func_class[func].next_invok_start:
                if left <= PRE_WARM and right >= -PRE_WARM:
                    memory.add(func)
                    func_class[func].load(0, 1)
                    break
        
        # 根据预测结果进行预热
        if func in pred_func_account:
            pred_result = pred_func_account[func]
            if pred_result[0] > 0:
                memory.add(func)
                func_class[func].load(0, pred_result[0])
    
    # 模拟
    # func_cold = {func: 0 for func in test_func_arrcount}    #冷启动次数
    # func_invok = {func: 0 for func in test_func_arrcount}   #调用次数
    # func_invok_seq = {func:[] for func in test_func_arrcount}

    # func_invok_bool = {func: 0 for func in test_func_arrcount}  #存在调用的单位时间
    # func_waste = {func: 0 for func in test_func_arrcount}   #内存浪费的单位时间
    test_func = set(test_func_arrcount.keys())
    func_cold = {func: 0 for func in predictable_func_ids & test_func}    #冷启动次数
    func_invok = {func: 0 for func in predictable_func_ids & test_func}   #调用次数
    func_invok_seq = {func:[] for func in predictable_func_ids & test_func}

    func_invok_bool = {func: 0 for func in predictable_func_ids & test_func}  #存在调用的单位时间
    func_waste = {func: 0 for func in predictable_func_ids & test_func}   #内存浪费的单位时间

    PID_controller = {}
    IAT_distribution = {}
    #########待取消
    # for func in test_func_arrcount:
    #     if func_class[func].type != REGULAR and func not in predictable_func_ids:    #除了Regular函数和可预测函数外的函数使用PID控制器
    #         PID_controller[func] = PID(0.6, 0.1, 0.05, setpoint=PID_TARGET)
    #         IAT_distribution[func] = distribution(HISTORY_TIMEOUT, HISTORY_LENGTH)
    #########待取消

    waste_mem_time = 0
    memory_size = []   

    with tqdm(total=1440 * 2) as pbar:
        for i in range(1440 * 2):
            # 每隔PREDICT_WINDOW分钟预测一次
            if i % PREDICT_WINDOW == PREDICT_WINDOW - 1 and i != 0:
                pbar.update(PREDICT_WINDOW)
                
                with open(f'/home/dell/xyf/AMC/method_exp/prediction_result/LazyProhet/{i+1}_{i+PREDICT_WINDOW}_result.pkl', 'rb') as f:
                    print(f'load {i+1}_{i+PREDICT_WINDOW}_result')
                    pred_func_account = pkl.load(f)    
        
                tasks = [(func, train_func_arrcount[func], func_invok_seq[func], LOCAL_WINDOW, PREDICT_WINDOW)
                        for func in predictable_func_ids if func in test_func_arrcount]                
                with Pool(2) as pool:
                    results = pool.map_async(predict_func, tasks)                    
                    pool.close()
                    pool.join()
                try:
                    all_results = results.get()
                    for func, pred_result in all_results:
                        if func is not None:
                            pred_func_account[func] = pred_result
                    
                    with open(f'./prediction_result/LazyProhet/{i+1}_{i+PREDICT_WINDOW}_result.pkl', 'wb') as f:
                        pkl.dump(pred_func_account, f)                    
                except Exception as e:
                    print(f"Error in proccessing: {e}")
    
            random.shuffle(func_corr_lst)
            for func in (func_lst + func_corr_lst): #In case of some functions staying in the memory forever
                #########待删
                # if func not in predictable_func_ids:
                #     continue
                # if func != 'e370dfc0407dec1f01bd6318b1da2433550d901d1410efc5be13d5125f81ab6d':    # Appro-regular-90mins  cold_rate:37/454    waster_memory:290
                #     continue
                if func != '72d7cec815a76ccb4a062882d5d3a651c5180a7578525173dc8efe2fc8452ee9':      # Appro-regular-240mins   cold_rate:4/16    waster_memory:119
                    continue
                #########待删
                _type = func_class[func].type

                # 模拟预热
                if i in func_class[func].containers_dict:   
                    load_num = func_class[func].containers_dict.pop(i)
                    func_class[func].set_containers(i, load_num)
                    if func_class[func].state:
                        memory.add(func)
                    else:
                        memory.remove(func) if func in memory else None

                if func in test_func_arrcount and test_func_arrcount[func][i] > 0: #invok
                    func_invok[func] += test_func_arrcount[func][i]
                    func_invok_seq[func].append(test_func_arrcount[func][i])
                    func_invok_bool[func] += 1
                    if func in IAT_distribution:
                        IAT_distribution[func].update(i)

                    if not func_class[func].state: #cold start
                        func_cold[func] += test_func_arrcount[func][i]
                        func_class[func].load(i, test_func_arrcount[func][i])
                        memory.add(func)
                    elif func_class[func].state and func_class[func].containers_num < test_func_arrcount[func][i]: # 部分冷启动
                        func_cold[func] += test_func_arrcount[func][i] - func_class[func].containers_num
                        func_class[func].load(i, test_func_arrcount[func][i]-func_class[func].containers_num)
                    elif func_class[func].state and func_class[func].containers_num >= test_func_arrcount[func][i]: # 热启动
                        waste_mem_time += func_class[func].containers_num - test_func_arrcount[func][i]

                    if func_class[func].wait_time is None or func_class[func].wait_time > 0: #A new invocation seq begins currently
                        func_class[func].pre_call_start = i
                        func_class[func].adp_wait.append(func_class[func].wait_time)

                        ## Shift: Update predictive values
                        if SHIFT and (_type in [REGULAR, DENSE, POSSIBLE, NEW_POSS, UNKNOWN]) and len(func_class[func].adp_wait) >= 5: # Adaptively Updating
                            if _type == REGULAR:
                                if abs(np.median(func_class[func].adp_wait) - func_class[func].pred_interval[0]) > max(1, func_class[func].idle_info["std"]):
                                    func_class[func].pred_interval[0] = (func_class[func].pred_interval[0]+np.median(func_class[func].adp_wait))/2

                        if SHIFT and len(func_invok_seq) % 180 == 178:    # 每3个小时更新一次可预测函数,预测前更新 
                            is_predicable = False
                            invoke_seq = np.array(func_invok_seq[func]).astype(float)
                            pe = permutation_entropy(func_invok_seq[func])
                            WTs = func_class[func].adp_wait
                            CV_WT = np.std(WTs) / np.mean(WTs)
                            if pe > 0.2:
                                is_predicable = True
                            elif CV_WT>2 and CV_WT<5 and pe>0.1:
                                is_predicable = True

                            if is_predicable:
                                predictable_func_ids.add(func)
                            else:
                                predictable_func_ids.discard(func)  
                                # PID_controller[func] = PID(0.6, 0.1, 0.5, setpoint=PID_TARGET)          # 不再可预测的函数使用PID控制器
                                # IAT_distribution[func] = distribution(HISTORY_TIMEOUT, HISTORY_LENGTH)

                    func_class[func].wait_time = 0
                    func_class[func].last_call = i
                    
                    #Update prediction for UNI, LIMITIDLE, DENSE, DIVI
                    if _type == REGULAR:
                        func_class[func].next_invok_start = [(i + 1 + func_class[func].pred_interval[0] - min(func_class[func].idle_info["std"], EN_STD),
                                                            i + 1 + func_class[func].pred_interval[0] + min(func_class[func].idle_info["std"], EN_STD))]
                            
                else: # not invok
                    #func_class[func].cal_idle()
                    if func_class[func].wait_time is None:
                        func_class[func].wait_time = 1
                    else:
                        func_class[func].wait_time += 1
                        
                    if func in memory:
                        waste_mem_time += func_class[func].containers_num
                        if func in func_waste:
                            func_waste[func] += 1
                    
                    # Prewarm according to different types
                    pre_warm_flag = False
                    
                    if _type == REGULAR:
                        (p_small, p_large) = func_class[func].next_invok_start[0]
                        p_small += (func_class[func].pred_interval[0] + 1) * int(p_large <= i) #allow for one missing hit
                        func_class[func].next_invok_start[0] = (p_small, p_small + 2 * min(func_class[func].idle_info["std"], EN_STD))
                        for (left, right) in func_class[func].next_invok_start:
                            if left <= i + PRE_WARM and right >= i - PRE_WARM:
                                pre_warm_flag = True
                                break
                        if func_class[func].state and (not pre_warm_flag) and func_class[func].wait_time >= GIVE_UP[_type]: #Remove
                            memory.remove(func)
                            func_class[func].unload()
                        elif pre_warm_flag:
                            memory.add(func)
                            func_class[func].load(i, 1)
                            
                if func in pred_func_account:     #可预测函数 
                    func_class[func].set_containers(i+1, pred_func_account[func][(i+1) % PREDICT_WINDOW])
                    if func_class[func].state:
                        memory.add(func)
                    elif not func_class[func].state and func in memory:
                        memory.remove(func)
                #########待取消
                # elif func in PID_controller:
                #     iat = IAT_distribution[func].predict_IAT(IAT_MIN, i, IAT_QUANTILE)
                #     func_cold_ratio = func_cold[func] / func_invok[func] if func_invok[func] > 0 else 0
                #     score = -1 * PID_controller[func](func_cold_ratio)
                #     if score < PID_TARGET * (1 - T_ALPHA / 2):    #冷启动率低于目标值，可以减少资源
                #         result = int(func_class[func].containers_num - max(1, func_class[func].containers_num * BETA))
                        
                #     elif score > PID_TARGET * (1 + T_ALPHA / 2):  #冷启动率高于目标值，需增加资源
                #         result = int(func_class[func].containers_num + max(1, func_class[func].containers_num * BETA))

                #     if result >= 0:
                #         func_class[func].containers_dict[i + iat] = result
                #     else:
                #         func_class[func].containers_dict[i + iat] = 0
                #########待取消
            memory_size.append(len(memory))
            
    cold_ratio = [cold/func_invok[func] for func, cold in func_cold.items()]
    print("WMT:", waste_mem_time, waste_mem_time/2880)
    print("Memory Usage", np.mean(memory_size), np.median(memory_size))
    print(np.percentile(cold_ratio, 50), np.percentile(cold_ratio, 75), np.percentile(cold_ratio, 90))

    #保存记录
    os.makedirs("./noConcurrency/PAC_result", exist_ok=True)
    cur_time = time.strftime("%m-%d-%H-%M", time.localtime())
    json_pretty_dump(func_cold, f"./noConcurrency/PAC_result/func_cold_{cur_time}.json")
    json_pretty_dump(func_waste, f"./noConcurrency/PAC_result/func_waste_{cur_time}.json")
    json_pretty_dump(func_invok, f"./noConcurrency/PAC_result/func_invok_{cur_time}.json")
    json_pretty_dump(memory_size, f"./noConcurrency/PAC_result/memory_size_{cur_time}.json")

    #分析
    cold_ratio = [cold/func_invok[func] for func, cold in func_cold.items()]
    aim = np.percentile(cold_ratio, 75)
    label_lst = ['Unknown','Warm', 'Regular', "Appro-regular", "Dense", "Successive", "Plused", "Possible", "Corr", "New_poss"]

    cold_num_per_type = [0] * TYPE_NUM
    invok_num_per_type = [0] * TYPE_NUM
    cold_ratio_per_type_lst = [[] for _ in range(TYPE_NUM)]

    for func, cold in func_cold.items():
        cold_num_per_type[func_class[func].type] += cold
        invok_num_per_type[func_class[func].type] += func_invok[func]
        cold_ratio_per_type_lst[func_class[func].type].append(cold / func_invok[func])

    cold_ratio_exceed_num, cold_ratio_exceed_rate, cold_ratio75 =[0]*TYPE_NUM, [0]*TYPE_NUM, [0]*TYPE_NUM

    for i, lst in enumerate(cold_ratio_per_type_lst):
        if len(lst) > 0:
            cold_ratio_exceed_num[i] = sum(np.where(np.array(lst)>=aim, 1, 0))
            cold_ratio_exceed_rate[i] = sum(np.where(np.array(lst)>=aim, 1, 0))/len(lst)
            cold_ratio75[i] = np.percentile(lst, 75)
        else:
            print(f"Empty Type {i}!")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10, 12))
    plt.subplots_adjust(wspace=0, hspace=0.3)

    ax1.bar(range(TYPE_NUM), 
            cold_num_per_type, 
            color=color_lst[:TYPE_NUM], 
            tick_label=label_lst
            )
    ax1.set_title('Cold Start Num')
    add_value_labels(ax1, cold_num_per_type)

    ax2.bar(range(TYPE_NUM), 
            invok_num_per_type, 
            color=color_lst[: TYPE_NUM], 
            tick_label=label_lst
            )
    ax2.set_title('Invocation Num')
    add_value_labels(ax2, invok_num_per_type)

    ax3.bar(range( TYPE_NUM), 
            cold_ratio_exceed_num, 
            color=color_lst[: TYPE_NUM],  
            tick_label=label_lst
            )
    ax3.set_title('Cold Start Exceed Num')
    add_value_labels(ax3, cold_ratio_exceed_num)

    ax4.bar(range( TYPE_NUM), 
            cold_ratio_exceed_rate, 
            color=color_lst[: TYPE_NUM],  
            tick_label=label_lst
            )
    ax4.set_title('Cold Start Exceed Ratio')
    add_value_labels(ax4, cold_ratio_exceed_rate)

    plt.savefig("./noConcurrency/PAC_result/PAC_result.png")