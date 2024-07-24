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
sys.path.append('/home/dell/xyf/azurefunctions-dataset2019/analysis')
sys.path.append('/home/dell/xyf/AMC')
from common import *
import polars as pl

if __name__ == "__main__": 

    # # 加载数据
    # file_name = "/home/dell/xyf/azure-data/invocations_per_function_md.anon.d"
    # train_file_names, test_file_names = [file_name+"%02d.csv" % (i) for i in range(1, 13)], [file_name+"13.csv", file_name+"14.csv"]

    # train_func_arrcount = {}    #函数负载数据
    # train_func_owner_app = {}   
    # train_owner_func = defaultdict(set)
    # train_app_func = defaultdict(set)

    # func_trigger = defaultdict(set)

    # for i, file in enumerate(train_file_names):
    #     df = pd.read_csv(file)
        
    #     for _, row in df.iterrows():
    #         func = row["HashFunction"]
    #         train_func_owner_app[func] = row["HashOwner"]+'\t'+row["HashApp"]
    #         train_owner_func[row["HashOwner"]].add(func)
    #         train_app_func[row["HashApp"]].add(func)
    #         func_trigger[func].add(row["Trigger"])
            
    #         if func not in train_func_arrcount:
    #             train_func_arrcount[func] = [0]*12*1440     # 空缺补0
    #         train_func_arrcount[func][i*1440: (i+1)*1440] = list(row[4:].values)
    #     del df

    # test_func_arrcount = {}
    # test_func_owner_app = {}
    # test_owner_func = defaultdict(set)
    # test_app_func = defaultdict(set)

    # for i, file in enumerate(test_file_names):
    #     df = pd.read_csv(file)
        
    #     for _, row in df.iterrows():
    #         func = row["HashFunction"]
    #         test_func_owner_app[func] = row["HashOwner"]+'\t'+row["HashApp"]
    #         test_owner_func[row["HashOwner"]].add(func)
    #         test_app_func[row["HashApp"]].add(func)
    #         func_trigger[func].add(row["Trigger"])
            
    #         if func not in test_func_arrcount:
    #             test_func_arrcount[func] = [0]*2*1440       # 空缺补0
    #         test_func_arrcount[func][i*1440: (i+1)*1440] = list(row[4:].values)
    #     del df
    # train_NUM, test_NUM = len(train_func_arrcount), len(test_func_arrcount)

    func_class = {}
    with open("/home/dell/xyf/azurefunctions-dataset2019/mid-data/train_info_assigned.txt") as rf:    # 所有函数的负载数据 hashID  forget  loadarray
        for line in rf:
            func, type, forget = line.strip().split('\t')
            func_class[func] = func_state(_type=int(type), forget=int(forget))
    print(len(func_class))

    train_candi = read_json("/home/dell/xyf/ColdStart/SPES/mid-data/func_candi.json")
    print(len(train_candi))

    candi_func_tuplelst = defaultdict(list)
    all_corr_candi_num = 0
    candi_dist = [0] * TYPE_NUM

    for func, tri_tuple_lst in train_candi.items():
        if func_class[func].type == CORR:
            func_all_corr_candi = True
        
            for (candi, lag, _) in tri_tuple_lst:
                if candi not in candi_func_tuplelst:
                    candi_dist[func_class[candi].type] += 1
                    
                candi_func_tuplelst[candi].append((func, lag))
                if func_class[candi].type != CORR:
                    func_all_corr_candi = False
        
            all_corr_candi_num += func_all_corr_candi

    # # 分类计算基本信息
    # additional_poss_num = 0

    # c = 0
    # with tqdm(total=train_NUM) as pbar:
    #     for func in train_func_arrcount:
    #         func_class[func].reset(True)
    #         c += 1
    #         if c % shown_func_num == 0:
    #             pbar.update(shown_func_num)
                
    #         if func_class[func].type == WARM: continue
                
    #         func_class[func].pred_interval = [] #pred_value

    #         arrcount = train_func_arrcount[func][1440*func_class[func].forget:]
    #         func_class[func].last_call = np.where(np.array(arrcount)>0)[0][-1] - len(arrcount)
    #         #assert func_class[func].last_call < 0

    #         invok = conj_seq_lst(arrcount, count_invoke=True)
    #         non_invok = conj_seq_lst(arrcount)

    #         func_class[func].wait_time = non_invok[-1] if arrcount[-1] == 0 else 0
    #         func_class[func].pre_call_start = func_class[func].last_call - invok[-1] + 1

    #         func_class[func].lasting_info = {"max": np.max(invok), "seq_num": len(invok)}
    #         func_class[func].idle_info = {"max": np.max(non_invok), "std": np.std(non_invok), "kind": len(set(non_invok))}
            
    #         if func_class[func].type == REGULAR:
    #             func_class[func].pred_interval = [np.median(non_invok)]
            
    #         else:
    #             non_invok_unique, non_invok_counts = np.unique(non_invok, return_counts=True)
    #             mode_cut = min(5, sum(np.where(non_invok_counts > 1, 1, 0)))
    #             ordered_index = np.argsort(-non_invok_counts)
    #             non_mode_max_index = ordered_index[: mode_cut]
                
    #             func_class[func].idle_info["mode"] = non_invok_unique[non_mode_max_index].tolist() #In the order of occuring count
    #             func_class[func].idle_info["mode_count"] = non_invok_counts[non_mode_max_index].tolist()

    #             if func_class[func].type == APPRO_REGULAR:
    #                 #func_class[func].pred_interval = non_invok_unique[ordered_index[: min(len(ordered_index), IDLE_NUM_MAX)]].tolist()
    #                 func_class[func].pred_interval = [non_invok_unique[idx] for idx in ordered_index[: min(len(ordered_index), IDLE_NUM_MAX)]
    #                                                                         if non_invok_counts[idx] > 0.1 * sum(non_invok_counts)]
                            
    #             elif func_class[func].type == DENSE:
    #                 if len(non_mode_max_index) > 0:
    #                     func_class[func].pred_interval = [min(func_class[func].idle_info["mode"]),
    #                         min(10, max(func_class[func].idle_info["mode"]))]
    #                 else:
    #                     func_class[func].pred_interval = [min(non_invok_unique), min(IDLE_NUM_MAX, max(non_invok_unique))]
                
    #             elif func_class[func].type == NEW_POSS:
    #                 func_class[func].type = 0
                
    #             if func_class[func].type == POSSIBLE or (func_class[func].type == UNKNOWN and len(non_mode_max_index) > 0):
    #                 additional_poss_num += (func_class[func].type == UNKNOWN)
    #                 func_class[func].type = POSSIBLE
    #                 if max(func_class[func].idle_info["mode"]) - min(func_class[func].idle_info["mode"]) <= DISCRETE_TH:
    #                     func_class[func].pred_interval = [min(func_class[func].idle_info["mode"]), 
    #                                                     max(func_class[func].idle_info["mode"])]
    #                     func_class[func].idle_info["pred_interval_discrete"] = False
    #                 else: 
    #                     func_class[func].pred_interval = sorted(list(func_class[func].idle_info["mode"]))
    #                     func_class[func].idle_info["pred_interval_discrete"] = True
    
    # # 加载测试集函数
    # func_lst, func_corr_lst = set(), set()
    # num_unseen_func = 0
    # for func in func_class:
    #     if func_class[func].type == CORR:
    #         func_corr_lst.add(func)
    #     else:
    #         func_lst.add(func)

    # for func in test_func_arrcount:
    #     if func in func_class: 
    #         continue
    #     num_unseen_func += 1
    #     func_lst.add(func)      # Unseen 函数 训练集中未出现的函数   
    #     func_class[func] = func_state() 
        

    # func_lst, func_corr_lst = list(func_lst), list(func_corr_lst)
    # print(len(func_class), len(func_lst)+len(func_corr_lst), len(test_func_arrcount), num_unseen_func)

    # test_func_corr = defaultdict(set)
    # for func, ownerapp in test_func_owner_app.items():
    #     if func not in train_func_arrcount:
    #         owner, app = ownerapp.split('\t')
    #         candi_func_set = (test_owner_func[owner] | test_app_func[app])
    #         if len(candi_func_set) == 1: continue
            
    #         candi_func_set.remove(func)
    #         for candi_func in candi_func_set:
    #             if len(func_trigger[func] & func_trigger[candi_func]) > 0:  #判断字符串是否相等
    #                 test_func_corr[func].add(candi_func)

    # test_func_corr_perform = {func: {candi_func: 0} for candi_func in test_func_corr[func]}
    
    # for func in test_func_arrcount:
    #     if func in func_class and (func not in train_func_arrcount):
    #         del func_class[func]
    #     if func not in func_class:
    #         func_class[func] = func_state()
    
    # c = 0
    # with tqdm(total=train_NUM) as pbar:
    #     for func in train_func_arrcount:
    #         func_class[func].reset(True)
    #         c += 1
    #         if c % shown_func_num == 0:
    #             pbar.update(shown_func_num)
                
    #         if func_class[func].type == WARM: continue
    #         arrcount = train_func_arrcount[func][1440*func_class[func].forget:]
    #         func_class[func].last_call = np.where(np.array(arrcount)>0)[0][-1] - len(arrcount)
    #         invok = conj_seq_lst(arrcount, count_invoke=True)
    #         non_invok = conj_seq_lst(arrcount)

    #         func_class[func].wait_time = non_invok[-1] if arrcount[-1] == 0 else 0
    #         func_class[func].pre_call_start = func_class[func].last_call - invok[-1] + 1

    # for func in train_func_arrcount: # calculate next invok start at time 0
    #     _type = func_class[func].type

    #     if _type == REGULAR: 
    #         p = func_class[func].last_call + 1
    #         if p < 0:
    #             p += (func_class[func].pred_interval[0] + 1)
    #         func_class[func].next_invok_start = [(p - min(func_class[func].idle_info["std"], EN_STD), 
    #                                             p + min(func_class[func].idle_info["std"], EN_STD))]

    #     elif _type == APPRO_REGULAR:
    #         func_class[func].next_invok_start = [(func_class[func].last_call + p + 1, func_class[func].last_call + p + 1) 
    #                                                 for p in func_class[func].pred_interval]
    #     elif _type == DENSE: 
    #         func_class[func].next_invok_start = [(func_class[func].last_call+func_class[func].pred_interval[0]+1,
    #                                             func_class[func].last_call+func_class[func].pred_interval[1]+1)]
    #     elif _type == CORR:
    #         func_class[func].next_invok_start = defaultdict(list)
    #         for (candi_func, lag, _ ) in train_candi[func]:
    #             func_class[func].next_invok_start[candi_func].append(
    #                 (func_class[candi_func].last_call, func_class[candi_func].last_call + lag)
    #             )
    #             if func_class[candi_func].type != CORR: #otherwise, endless loop
    #                 func_class[func].next_invok_start[candi_func] += [(nis[0], nis[1] + lag) for nis in func_class[candi_func].next_invok_start]

    #     elif _type in [POSSIBLE, NEW_POSS]:
    #         if func_class[func].idle_info["pred_interval_discrete"]:
    #             func_class[func].next_invok_start = [(func_class[func].last_call + p + 1, func_class[func].last_call + p + 1) 
    #                                                     for p in func_class[func].pred_interval]
    #         else:
    #             func_class[func].next_invok_start = [(func_class[func].last_call+func_class[func].pred_interval[0]+1,
    #                                             func_class[func].last_call+func_class[func].pred_interval[1]+1)]

    with open('/home/dell/xyf/AMC/variables_checkpoint.pkl', "rb") as file:
        train_func_ids, test_func_arrcount, func_class, func_lst, func_corr_lst, test_func_corr, test_func_corr_perform= pkl.load(file)
    
    memory = set()

    for func in train_func_ids: #Pre_warm for testing at time 0
        _type = func_class[func].type
        
        if _type == WARM:
            memory.add(func)
            func_class[func].load(0,1)

        elif _type in [SUCCESSIVE, CORR, POSSIBLE, NEW_POSS, UNKNOWN] and (func_class[func].last_call == -1):            
            memory.add(func)
            func_class[func].load(0,1)
        
        elif _type == PLUSED and (func_class[func].last_call + GIVE_UP[PLUSED] >= 0):
            memory.add(func)
            func_class[func].load(0,1)

        elif _type in [REGULAR, APPRO_REGULAR, DENSE] and func_class[func].last_call == -1 \
            and (func_class[func].lasting_info["max"] > 1 or func_class[func].lasting_info["seq_num"] == 1):
            memory.add(func)
            func_class[func].load(0,1)

        # Pre-warm funcs having next_invok_start
        if _type == CORR:
            flag = False
            for _, tuple_lst in func_class[func].next_invok_start.items():
                for (left, right) in tuple_lst:
                    if left <= PRE_WARM and right >= -PRE_WARM:
                        memory.add(func)
                        func_class[func].load(0,1)
                        flag = True
                        break
                if flag: break
                    
        elif _type in [REGULAR, APPRO_REGULAR, DENSE, POSSIBLE, NEW_POSS]:
            for (left, right) in func_class[func].next_invok_start:
                if left <= PRE_WARM and right >= -PRE_WARM:
                    memory.add(func)
                    func_class[func].load(0,1)
                    break

    #########待删
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
    #########待删

    # 模拟
    # func_cold = {func: 0 for func in test_func_arrcount}
    # func_invok = {func: 0 for func in test_func_arrcount}

    # func_invok_bool = {func: 0 for func in test_func_arrcount}
    # func_waste = {func: 0 for func in test_func_arrcount}
    test_func = set(test_func_arrcount.keys())
    func_cold = {func: 0 for func in predictable_func_ids & test_func}    #冷启动次数
    func_invok = {func: 0 for func in predictable_func_ids & test_func}   #调用次数

    func_invok_bool = {func: 0 for func in predictable_func_ids & test_func}  #存在调用的单位时间
    func_waste = {func: 0 for func in predictable_func_ids & test_func}   #内存浪费的单位时间

    waste_mem_time = 0
    memory_size = []   

    with tqdm(total=1440 * 2) as pbar:
        for i in range(1440 * 2):
            if (i+1) % 40 == 0:  pbar.update(40)
            
            random.shuffle(func_corr_lst)
            for func in (func_lst + func_corr_lst): #In case of some functions staying in the memory forever
                #########待删
                # if func not in predictable_func_ids:
                #     continue
                # if func != 'e370dfc0407dec1f01bd6318b1da2433550d901d1410efc5be13d5125f81ab6d':    # Appro-regular-90min  cold_rate:18/454  waste_memory:1248
                #     continue
                if func != '72d7cec815a76ccb4a062882d5d3a651c5180a7578525173dc8efe2fc8452ee9':  #Appro-regular-240mins   cold_rate:2/16    waster_memory:143
                    continue
                #########待删
                _type = func_class[func].type
                
                if func in test_func_arrcount and test_func_arrcount[func][i] > 0:  # invok
                    func_invok[func] += test_func_arrcount[func][i]
                    func_invok_bool[func] += 1
                    
                    if not func_class[func].state: # cold start 无实例完全冷启动
                        # func_cold[func] += 1      # SPES的做法，冷启动次数不计入并发度
                        func_cold[func] += test_func_arrcount[func][i]  
                        func_class[func].load(i, test_func_arrcount[func][i])   
                        memory.add(func)
                    elif func_class[func].state and func_class[func].containers_num < test_func_arrcount[func][i]: # 部分冷启动
                        func_cold[func] += test_func_arrcount[func][i] - func_class[func].containers_num
                        func_class[func].load(i, test_func_arrcount[func][i]-func_class[func].containers_num)

                    elif func_class[func].state and func_class[func].containers_num >= test_func_arrcount[func][i]: # 热启动
                        waste_mem_time += func_class[func].containers_num - test_func_arrcount[func][i]
                        

                    if func_class[func].wait_time is None or func_class[func].wait_time > 0: # A new invocation seq begins currently
                        func_class[func].pre_call_start = i
                        func_class[func].adp_wait.append(func_class[func].wait_time)
                        
                        ## Shift: Update predictive values
                        if SHIFT and (_type in [REGULAR, DENSE, POSSIBLE, NEW_POSS, UNKNOWN]) and len(func_class[func].adp_wait) >= 5: # Adaptively Updating
                            if _type == REGULAR:
                                if abs(np.median(func_class[func].adp_wait) - func_class[func].pred_interval[0]) > max(1, func_class[func].idle_info["std"]):
                                    func_class[func].pred_interval[0] = (func_class[func].pred_interval[0]+np.median(func_class[func].adp_wait))/2
                                    
                            elif _type == DENSE:
                                if abs(min(func_class[func].adp_wait) - func_class[func].pred_interval[0]) > 1:
                                    func_class[func].pred_interval[0] = min(func_class[func].adp_wait)
                                if (abs(max(func_class[func].adp_wait) - func_class[func].pred_interval[1]) > 1) \
                                    and max(func_class[func].adp_wait) <= DENSE_UPPER_BOUND:
                                    func_class[func].pred_interval[1] = max(func_class[func].adp_wait)
                            
                            elif _type in [POSSIBLE, NEW_POSS]:
                                idle_unique, idle_counts = np.unique(func_class[func].adp_wait, return_counts=True)
                                if max(idle_counts) >= max(2, func_class[func].idle_info["mode_count"][-1]):
                                    idle_max_index = np.argsort(-idle_counts)
                                    for idx in idle_max_index:
                                        if idle_unique[idx] in func_class[func].idle_info["mode"]: continue
                                        pos = len(func_class[func].idle_info["mode_count"]) - \
                                                bisect.bisect(func_class[func].idle_info["mode_count"][::-1], idle_counts[idx])
                                        if pos >= len(func_class[func].idle_info["mode_count"]): continue
                                            
                                        if idle_counts[idx] > func_class[func].idle_info["mode_count"][pos]:
                                            func_class[func].idle_info["mode_count"][pos] = idle_counts[idx]
                                            func_class[func].idle_info["mode"][pos] = idle_unique[idx]
                                        elif idle_counts[idx] == func_class[func].idle_info["mode_count"][pos]:
                                            func_class[func].idle_info["mode_count"].insert(pos, idle_counts[idx])
                                            func_class[func].idle_info["mode"].insert(pos, idle_unique[idx])

                                    if max(func_class[func].idle_info["mode"]) - min(func_class[func].idle_info["mode"]) <= DISCRETE_TH:
                                        func_class[func].pred_interval = [min(func_class[func].idle_info["mode"]), 
                                                        max(func_class[func].idle_info["mode"])]
                                        func_class[func].idle_info["pred_interval_discrete"] = False
                                    else: 
                                        func_class[func].pred_interval = sorted(list(func_class[func].idle_info["mode"]))
                                        func_class[func].idle_info["pred_interval_discrete"] = True
                                    
                            elif _type == UNKNOWN:
                                idle_unique, idle_counts = np.unique(func_class[func].adp_wait, return_counts=True)
                                idxs = np.argsort(-idle_counts)[: min(5, sum(np.where(idle_counts > 1, 1, 0)))]
                                mode_lst = idle_unique[idxs]
                                if len(mode_lst) > 0:
                                    func_class[func].type = NEW_POSS
                                    func_class[func].idle_info["mode"] = mode_lst.tolist()
                                    func_class[func].idle_info["mode_count"] = idle_counts[idxs].tolist()
                                    
                                    if max(mode_lst) - min(mode_lst) <= DISCRETE_TH:
                                        func_class[func].idle_info["pred_interval_discrete"] = False
                                        func_class[func].pred_interval = [min(mode_lst), max(mode_lst)]
                                    else:
                                        func_class[func].idle_info["pred_interval_discrete"] = True
                                        func_class[func].pred_interval = sorted(list(mode_lst))
                                    
                    func_class[func].wait_time = 0
                    func_class[func].last_call = i
                    
                    #Update prediction for UNI, LIMITIDLE, DENSE, DIVI
                    if _type == REGULAR:
                        func_class[func].next_invok_start = [(i + 1 + func_class[func].pred_interval[0] - min(func_class[func].idle_info["std"], EN_STD),
                                                            i + 1 + func_class[func].pred_interval[0] + min(func_class[func].idle_info["std"], EN_STD))]
                            
                    elif _type == APPRO_REGULAR or (_type in [POSSIBLE, NEW_POSS] and func_class[func].idle_info["pred_interval_discrete"]):
                        func_class[func].next_invok_start = [(i + p + 1, i + p + 1) for p in func_class[func].pred_interval]
                    
                    elif _type == DENSE or (_type in [POSSIBLE, NEW_POSS] and (not func_class[func].idle_info["pred_interval_discrete"])):
                        func_class[func].next_invok_start = [(i + func_class[func].pred_interval[0] + 1, i + func_class[func].pred_interval[1] + 1)]
                    
                    #The occurrences of candidate functions can be predictive indicators.

                    if func in candi_func_tuplelst: 
                        for (target_func, lag) in candi_func_tuplelst[func]: #func is candi, target is target
                            func_class[target_func].next_invok_start[func] = [(i, i + lag)]
                            if _type in [REGULAR, APPRO_REGULAR, DENSE, POSSIBLE, NEW_POSS]:
                                func_class[target_func].next_invok_start[func] += [(l, r + lag) 
                                                                                for (l, r) in func_class[func].next_invok_start]
                    if func in test_func_corr:
                        for target_func in test_func_corr[func]:
                            if func_class[target_func].type == CORR:
                                func_class[target_func].next_invok_start[func] += [(i, i + 1)]
                                func_class[target_func].next_invok_start[func] = list(set(func_class[target_func].next_invok_start[func]))
                            else:
                                #assert not isinstance(func_class[post_func].next_invok_start, dict), post_func
                                func_class[target_func].next_invok_start += [(i, i + 1)]
                                func_class[target_func].next_invok_start = list(set(func_class[target_func].next_invok_start))
                            
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

                    if _type in [REGULAR, APPRO_REGULAR, DENSE, POSSIBLE, NEW_POSS, UNKNOWN]:
                        for (left, right) in func_class[func].next_invok_start:
                            if left <= i + PRE_WARM and right >= i - PRE_WARM:
                                pre_warm_flag = True
                                break

                    elif _type == CORR:
                        for tuple_lst in func_class[func].next_invok_start.values():
                            for (left, right) in tuple_lst:
                                if left <= i + PRE_WARM and right >= i - PRE_WARM:
                                    pre_warm_flag = True
                                    break
                            if pre_warm_flag: break
                    
                    if func_class[func].state and (not pre_warm_flag) and func_class[func].wait_time >= GIVE_UP[_type]: #Remove
                        memory.remove(func)
                        func_class[func].unload()
                    
                    elif (not func_class[func].state) and pre_warm_flag:
                        memory.add(func)
                        func_class[func].load(i, 1)     #原版的预热：只预热一个容器
            size = 0
            for func in memory:
                size += func_class[func].containers_num
            memory_size.append(size)
            
            ## Update test_corr
            for func, perform_dict in test_func_corr_perform:
                for candi in perform_dict:
                    test_func_corr_perform[func][candi] += int(bool(test_func_arrcount[func][i]) & bool(test_func_arrcount[candi][i]))
            
            for func, perform_dict in test_func_corr_perform.items():
                best_perform = max(perform_dict.values())
                del_lst = []
                for candi, perform in perform_dict:
                    if best_perform - perform >= CORR_REMOVAL_TH:
                        del_lst.append(candi)
                for candi in del_lst:
                    del test_func_corr[func][candi]
                    del test_func_corr_perform[func][candi]
            
    cold_ratio = [cold/func_invok[func] for func, cold in func_cold.items()]
    print("WMT:", waste_mem_time, waste_mem_time/2880)
    print("Memory Usage", np.mean(memory_size), np.median(memory_size))
    print(np.percentile(cold_ratio, 50), np.percentile(cold_ratio, 75), np.percentile(cold_ratio, 90))

    #保存记录
    os.makedirs("./noConcurrency/SPES_result", exist_ok=True)
    cur_time = time.strftime("%m-%d-%H-%M", time.localtime())
    json_pretty_dump(func_cold, f"./noConcurrency/SPES_result/func_cold_{cur_time}.json")
    json_pretty_dump(func_waste, f"./noConcurrency/SPES_result/func_waste_{cur_time}.json")
    json_pretty_dump(func_invok, f"./noConcurrency/SPES_result/func_invok_{cur_time}.json")
    json_pretty_dump(memory_size, f"./noConcurrency/SPES_result/memory_size_{cur_time}.json")


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

    plt.savefig("./noConcurrency/SPES_result/SPES_result.png")