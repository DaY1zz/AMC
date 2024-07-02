import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pickle as pkl
sys.path.append('/home/dell/xyf/AMC')
from common import *
import types
import polars as pl

def is_picklable(obj):
    try:
        pkl.dumps(obj)
        return True
    except (pkl.PicklingError, TypeError):
        return False
    
def save_checkpoint(filename, variables):
    with open(filename, 'wb') as f:
        filtered_variables = {k: v for k, v in variables.items() if is_picklable(v)}
        pkl.dump(filtered_variables, f)

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

    # func_class = {}
    # with open("/home/dell/xyf/azurefunctions-dataset2019/mid-data/train_info_assigned.txt") as rf:    # 所有函数的负载数据 hashID  forget  loadarray
    #     for line in rf:
    #         func, type, forget = line.strip().split('\t')
    #         func_class[func] = func_state(_type=int(type), forget=int(forget))
    # print(len(func_class))

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
    
    # variables = globals().copy()
    # variables = {k: v for k, v in variables.items() if not k.startswith('__') and not callable(v)}
    # save_checkpoint('var_checkpoint.pkl', variables)

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
    print(len(predictable_func_ids))
    pred_func_account = {}
    # LighGBM 的参数
    boosting_params = {
                            "objective": "regression",
                            "metric": "mape",
                            "verbosity": -1,
                            "boosting_type": "gbdt",
                            "seed": 42,
                            "learning_rate": 0.1,
                            "min_child_samples": 4,
                            "num_leaves": 128,
                            "num_iterations": 100
                                        }
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
    
    # with open('pre_pred_result.pkl', 'wb') as f:
    #     pkl.dump(pred_func_account, f)

