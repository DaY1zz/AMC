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

    # 加载数据
    file_name = "/home/dell/xyf/azure-data/invocations_per_function_md.anon.d"
    train_file_names, test_file_names = [file_name+"%02d.csv" % (i) for i in range(1, 13)], [file_name+"13.csv", file_name+"14.csv"]

    train_func_arrcount = {}    #函数负载数据
    train_func_owner_app = {}   
    train_owner_func = defaultdict(set)
    train_app_func = defaultdict(set)

    func_trigger = defaultdict(set)

    for i, file in enumerate(train_file_names):
        df = pd.read_csv(file)
        
        for _, row in df.iterrows():
            func = row["HashFunction"]
            train_func_owner_app[func] = row["HashOwner"]+'\t'+row["HashApp"]
            train_owner_func[row["HashOwner"]].add(func)
            train_app_func[row["HashApp"]].add(func)
            func_trigger[func].add(row["Trigger"])
            
            if func not in train_func_arrcount:
                train_func_arrcount[func] = [0]*12*1440     # 空缺补0
            train_func_arrcount[func][i*1440: (i+1)*1440] = list(row[4:].values)
        del df

    test_func_arrcount = {}
    test_func_owner_app = {}
    test_owner_func = defaultdict(set)
    test_app_func = defaultdict(set)

    for i, file in enumerate(test_file_names):
        df = pd.read_csv(file)
        
        for _, row in df.iterrows():
            func = row["HashFunction"]
            test_func_owner_app[func] = row["HashOwner"]+'\t'+row["HashApp"]
            test_owner_func[row["HashOwner"]].add(func)
            test_app_func[row["HashApp"]].add(func)
            func_trigger[func].add(row["Trigger"])
            
            if func not in test_func_arrcount:
                test_func_arrcount[func] = [0]*2*1440       # 空缺补0
            test_func_arrcount[func][i*1440: (i+1)*1440] = list(row[4:].values)
        del df
    train_NUM, test_NUM = len(train_func_arrcount), len(test_func_arrcount)

    func_class = {}
    with open("/home/dell/xyf/azurefunctions-dataset2019/mid-data/train_info_assigned.txt") as rf:    # 所有函数的负载数据 hashID  forget  loadarray
        for line in rf:
            func, type, forget = line.strip().split('\t')
            func_class[func] = func_state(_type=int(type), forget=int(forget))
    print(len(func_class))

    # 分类计算基本信息
    additional_poss_num = 0
    c = 0
    with tqdm(total=train_NUM) as pbar:
        for func in train_func_arrcount:
            func_class[func].reset(True)
            c += 1
            if c % shown_func_num == 0:
                pbar.update(shown_func_num)
                
            if func_class[func].type == WARM: continue
                
            func_class[func].pred_interval = [] #pred_value

            arrcount = train_func_arrcount[func][1440*func_class[func].forget:]
            func_class[func].last_call = np.where(np.array(arrcount)>0)[0][-1] - len(arrcount)
            #assert func_class[func].last_call < 0

            invok = conj_seq_lst(arrcount, count_invoke=True)
            non_invok = conj_seq_lst(arrcount)

            func_class[func].wait_time = non_invok[-1] if arrcount[-1] == 0 else 0
            func_class[func].pre_call_start = func_class[func].last_call - invok[-1] + 1

            func_class[func].lasting_info = {"max": np.max(invok), "seq_num": len(invok)}
            func_class[func].idle_info = {"max": np.max(non_invok), "std": np.std(non_invok), "kind": len(set(non_invok))}
            
            if func_class[func].type == REGULAR:
                func_class[func].pred_interval = [np.median(non_invok)]
    
    # 加载测试集函数
    func_lst, func_corr_lst = set(), set()
    num_unseen_func = 0
    for func in func_class:
        if func_class[func].type == CORR:
            func_corr_lst.add(func)
        else:
            func_lst.add(func)

    for func in test_func_arrcount:
        if func in func_class: 
            continue
        num_unseen_func += 1
        func_lst.add(func)      # Unseen 函数 训练集中未出现的函数   
        func_class[func] = func_state() 
        

    func_lst, func_corr_lst = list(func_lst), list(func_corr_lst)
    print(len(func_class), len(func_lst)+len(func_corr_lst), len(test_func_arrcount), num_unseen_func)

    test_func_corr = defaultdict(set)
    for func, ownerapp in test_func_owner_app.items():
        if func not in train_func_arrcount:
            owner, app = ownerapp.split('\t')
            candi_func_set = (test_owner_func[owner] | test_app_func[app])
            if len(candi_func_set) == 1: continue
            
            candi_func_set.remove(func)
            for candi_func in candi_func_set:
                if len(func_trigger[func] & func_trigger[candi_func]) > 0:  #判断字符串是否相等
                    test_func_corr[func].add(candi_func)

    test_func_corr_perform = {func: {candi_func: 0} for candi_func in test_func_corr[func]}
    
    for func in test_func_arrcount:
        if func in func_class and (func not in train_func_arrcount):
            del func_class[func]
        if func not in func_class:
            func_class[func] = func_state()
    
    c = 0
    with tqdm(total=train_NUM) as pbar:
        for func in train_func_arrcount:
            func_class[func].reset(True)
            c += 1
            if c % shown_func_num == 0:
                pbar.update(shown_func_num)
                
            if func_class[func].type == WARM: continue
            arrcount = train_func_arrcount[func][1440*func_class[func].forget:]
            func_class[func].last_call = np.where(np.array(arrcount)>0)[0][-1] - len(arrcount)
            invok = conj_seq_lst(arrcount, count_invoke=True)
            non_invok = conj_seq_lst(arrcount)

            func_class[func].wait_time = non_invok[-1] if arrcount[-1] == 0 else 0
            func_class[func].pre_call_start = func_class[func].last_call - invok[-1] + 1

    for func in train_func_arrcount: # calculate next invok start at time 0
        _type = func_class[func].type

        if _type == REGULAR: 
            p = func_class[func].last_call + 1
            if p < 0:
                p += (func_class[func].pred_interval[0] + 1)
            func_class[func].next_invok_start = [(p - min(func_class[func].idle_info["std"], EN_STD), 
                                                p + min(func_class[func].idle_info["std"], EN_STD))]
    
    # 筛选出可预测函数并根据训练集进行初步预测
    PE_THRESHOLD = 0.2

    CV_WT_UPPER_THRESHOLD = 5
    CV_WT_LOWER_THRESHOLD = 2

    LOCAL_WINDOW = 60*48
    PREDICT_WINDOW = 60

    PID_TARGET = 0.15   #冷启动率
    T_ALPHA = 0.2

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
    #         pred_func_account[func] = pred_result
    #         pbar.update(1)
    with open('./prediction_result/LazyProhet/0_59_result.pkl', 'rb') as f:
        pred_func_account = pkl.load(f)

    memory = set()

    for func in train_func_arrcount: #Pre_warm for testing at time 0
        _type = func_class[func].type

        if _type == WARM:
            memory.add(func)
            func_class[func].load(0)
        elif _type == REGULAR and func_class[func].last_call == -1 \
            and (func_class[func].lasting_info["max"] > 1 or func_class[func].lasting_info["seq_num"] == 1):
            memory.add(func)
            func_class[func].load(0)
        if _type == REGULAR:
            for (left, right) in func_class[func].next_invok_start:
                if left <= PRE_WARM and right >= -PRE_WARM:
                    memory.add(func)
                    func_class[func].load(0)
                    break
        
        # 根据预测结果进行预热
        if func in pred_func_account:
            pred_result = pred_func_account[func]
            if not func_class[func].state and pred_result[0] > 0:
                memory.add(func)
                func_class[func].load(0)
    
    # 模拟
    func_cold = {func: 0 for func in test_func_arrcount}    #冷启动次数
    func_invok = {func: 0 for func in test_func_arrcount}   #调用次数
    func_invok_seq = {func:[] for func in test_func_arrcount}

    func_invok_bool = {func: 0 for func in test_func_arrcount}  #存在调用的单位时间
    func_waste = {func: 0 for func in test_func_arrcount}   #内存浪费的单位时间

    PID_controller = {}
    for func in test_func_arrcount:
        if func_class[func].type not in (REGULAR,WARM) and func not in predictable_func_ids:    #除了Regular函数和可预测函数外的函数使用PID控制器
            PID_controller[func] = PID(0.6, 0.1, 0.05, setpoint=PID_TARGET)

    waste_mem_time = 0
    memory_size = []   

    with tqdm(total=1440 * 2) as pbar:
        for i in range(1440 * 2):
            # 每隔PREDICT_WINDOW分钟预测一次
            if i % PREDICT_WINDOW == PREDICT_WINDOW - 1 and i != 0:
                pbar.update(PREDICT_WINDOW)
                with open(f'./prediction_result/LazyProhet/{i+1}_{i+PREDICT_WINDOW}_result.pkl', 'rb') as f:
                    pred_func_account = pkl.load(f)                
                # tasks = [(func, train_func_arrcount[func], func_invok_seq[func], LOCAL_WINDOW, PREDICT_WINDOW)
                #         for func in predictable_func_ids if func in test_func_arrcount]                
                # with Pool(2) as pool:
                #     results = pool.map_async(predict_func, tasks)                    
                #     pool.close()
                #     pool.join()
                # try:
                #     all_results = results.get()
                #     for func, pred_result in all_results:
                #         if func is not None:
                #             pred_func_account[func] = pred_result
                    
                #     with open(f'{i+1}_{i+PREDICT_WINDOW}_result.pkl', 'wb') as f:
                #         pkl.dump(pred_func_account, f)                    
                # except Exception as e:
                #     print(f"Error in proccessing: {e}")
    
            random.shuffle(func_corr_lst)
            for func in (func_lst + func_corr_lst): #In case of some functions staying in the memory forever
                _type = func_class[func].type
                
                if func in test_func_arrcount and test_func_arrcount[func][i] > 0: #invok
                    func_invok[func] += test_func_arrcount[func][i]
                    func_invok_seq[func].append(test_func_arrcount[func][i])
                    func_invok_bool[func] += 1
                    
                    if not func_class[func].state: #cold start
                        func_cold[func] += test_func_arrcount[func][i]
                        func_class[func].load(i)
                        memory.add(func)
                        
                    if func_class[func].wait_time is None or func_class[func].wait_time > 0: #A new invocation seq begins currently
                        func_class[func].pre_call_start = i
                        func_class[func].adp_wait.append(func_class[func].wait_time)

                        ## Shift: Update predictive values
                        if SHIFT and (_type in [REGULAR, DENSE, POSSIBLE, NEW_POSS, UNKNOWN]) and len(func_class[func].adp_wait) >= 5: # Adaptively Updating
                            if _type == REGULAR:
                                if abs(np.median(func_class[func].adp_wait) - func_class[func].pred_interval[0]) > max(1, func_class[func].idle_info["std"]):
                                    func_class[func].pred_interval[0] = (func_class[func].pred_interval[0]+np.median(func_class[func].adp_wait))/2
                        if SHIFT and len(func_invok_seq) % 360 == 358:    # 每6个小时更新一次可预测函数,预测前更新 
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
                                PID_controller[func] = PID(0.6, 0.1, 0.5, setpoint=PID_TARGET)    #不再可预测的函数使用PID控制器

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
                        waste_mem_time += 1
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
                            func_class[func].load(i)
                            
                    elif func in pred_func_account:     #可预测函数 TODO：请求并发度
                        if pred_func_account[func][(i+1) % PREDICT_WINDOW] > 0:
                            pre_warm_flag = True
                        if _type != WARM and func_class[func].state and (not pre_warm_flag):
                            memory.remove(func)
                            func_class[func].unload()
                        elif pre_warm_flag:
                            memory.add(func)
                            func_class[func].load(i)

                    elif func in PID_controller:
                        func_cold_ratio = func_cold[func] / func_invok[func] if func_invok[func] > 0 else 0
                        score = -1 * PID_controller[func](func_cold_ratio)
                        if score < PID_TARGET * (1 - T_ALPHA / 2) and func_class[func].state:    #冷启动率低于目标值，可以减少资源
                            memory.remove(func)
                            func_class[func].unload()
                        elif score > PID_TARGET * (1 + T_ALPHA / 2) and not func_class[func].state:  #冷启动率高于目标值，需增加资源
                            memory.add(func)
                            func_class[func].load(i)                           

            memory_size.append(len(memory))
            
    cold_ratio = [cold/func_invok[func] for func, cold in func_cold.items()]
    print("WMT:", waste_mem_time, waste_mem_time/2880)
    print("Memory Usage", np.mean(memory_size), np.median(memory_size))
    print(np.percentile(cold_ratio, 50), np.percentile(cold_ratio, 75), np.percentile(cold_ratio, 90))

    #保存记录
    os.makedirs("./PAC_result", exist_ok=True)
    cur_time = time.strftime("%m-%d-%H-%M", time.localtime())
    json_pretty_dump(func_cold, f"./PAC_result/func_cold_{cur_time}.json")
    json_pretty_dump(func_waste, f"./PAC_result/func_waste_{cur_time}.json")
    json_pretty_dump(func_invok, f"./PAC_result/func_invok_{cur_time}.json")
    json_pretty_dump(memory_size, f"./PAC_result/memory_size_{cur_time}.json")

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

    plt.savefig("./PAC_result/PAC_result.png")