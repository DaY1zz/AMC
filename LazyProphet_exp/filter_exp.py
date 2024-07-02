from LazyProphet import LazyProphet as lp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle as pkl
import polars as pl
import pandas as pd
import gc
from collections import defaultdict
import sys
sys.path.append('/home/dell/xyf/azurefunctions-dataset2019/analysis')

PROCESSED_12DAYS_DIR = "/home/dell/xyf/azurefunctions-dataset2019/processed/12days/"
DATASET_LENGTH = 12
valid_split_DAY = 9
label_lst = ['Unknown','Warm', 'Regular', "Appro-regular", "Dense", "Successive", "Plused", "Possible", "Corr"]

def conj_seq_lst(lst, count_invoke=False, threshold=1):
    seq_lenth_lst = []
    pre_pos = -1
    for i, e in enumerate(lst):
        if not (bool(e) ^ count_invoke): #非异或 判断两个条件是否相等 (求连续正且当前元素为正 或 求连续负且当前元素为负)
            if pre_pos < 0:     #连续序列中的第一个元素位置
                pre_pos = i
            if i == len(lst)-1 and i+1-pre_pos >= threshold:    #末尾元素进行处理
                seq_lenth_lst.append(i+1-pre_pos)
        else:   # 连续序列中断
            if pre_pos>=0 and i-pre_pos >= threshold:
                seq_lenth_lst.append(i-pre_pos)
            pre_pos = -1
    return seq_lenth_lst

class func_state:
    def __init__(self, _type = 0, forget = 0):
        self.type = _type
        self.forget = forget
        
        self.state = False # loaded or not
        self.load_time = None 
        self.wait_time = None 
        self.last_call = None
        self.pre_call_start = None # start of the last calling series
        
        self.idle_info = {} # "mode"：WT mode、 "mode_count": mode 出现次数
        self.invok_info = {}
        self.lasting_info = {}  #

        self.pred_interval = [] # 预测值
        self.pred_value = []
        self.next_invok_start = []
        
        self.adp_wait = []
    
    def load(self, load_time):
        self.state = True
        self.load_time = load_time
    
    def cal_lasting(self, cur_time):
        if not self.state:
            return 0
        return cur_time - self.load_time + 1
    
    def unload(self):
        self.state = False
        self.load_time = None
    
    def cal_wait(self):
        if self.wait_time is None:
            self.wait_time = 0
        self.wait_time += 1
    
    def reset(self, pred=False):
        self.unload()
        self.wait_time = None 
        self.last_call = None
        self.pre_call_start = None
        
        self.adp_wait = []
        
        if pred:
            self.next_invok_start = []

if __name__ == "__main__": 
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

    func_class = {}
    with open("/home/dell/xyf/azurefunctions-dataset2019/mid-data/train_info_assigned.txt") as rf:    # 所有函数的负载数据 hashID  forget  loadarray
        for line in rf:
            func, type, forget = line.strip().split('\t')
            func_class[func] = func_state(_type=type, forget=forget)

    df = pl.read_csv("func_info.csv")
    # cv_WT_result = []
    # for func, arrcount in train_func_arrcount.items():
    #     WTs = conj_seq_lst(arrcount)
    #     CV_WT = np.std(WTs) / np.mean(WTs)
    #     cv_WT_result.append((func, CV_WT))
    # df_cv_WT = pl.DataFrame(cv_WT_result, schema=['Function', 'CV_WT'],orient="row")
    # print(df_cv_WT)

    # df_cv_WT.write_csv("cv_WT.csv")

    CV_UPPER_THRESHOLD = 5
    CV_LOWER_THRESHOLD = 0.1
    PE_THRESHOLD = 0.2
    PERIOD_STRENGTH_THRESHOLD = 0.3

    CV_WT_UPPER_THRESHOLD = 5
    CV_WT_LOWER_THRESHOLD = 2

    BATCH_SIZE = 100

    df = df.filter(pl.col('Type') != 2)   #过滤regular

    pe_df = df.filter(pl.col('PE') > PE_THRESHOLD)
    cv_WT_df = df.filter((pl.col('CV_WT') > CV_WT_LOWER_THRESHOLD))\
            .filter((pl.col('CV_WT') < CV_WT_UPPER_THRESHOLD))\
            .filter(~pl.col('CV_WT').is_nan())\
            .filter(pl.col('PE')> 0.1)\
            
    df_union = pl.concat([pe_df, cv_WT_df]).unique()
    print(df_union)
    
    func_ids = df_union.select('Function').to_numpy().flatten()
    for func in tqdm(func_ids, desc="Processing functions"):
        arr = train_func_arrcount[func]
        plt.figure(figsize=(16,8))
        x1 = [i for i in range(1, len(arr) + 1)]
        
        plt.plot(x1, arr, color="blue")
        func_info = df_union.filter(pl.col('Function') == func).to_dict(as_series=False)

        # print(func+'\t'+'Type:{}'.format(label_lst[int(func_info['Type'][0])]))
        # print("CV:{}\tPE:{}\tPeriod:{}\tPeriod_Strength:{}"
        #     .format(func_info['CV'][0],func_info['PE'][0],func_info['Period'][0],func_info['Period_Strength'][0]))
        cv = round(func_info["CV"][0],4)
        pe = round(func_info["PE"][0],4)
        period = round(func_info["Period"][0],4)
        ps = round(func_info["Period_Strength"][0],4)
        cv_WT = round(func_info["CV_WT"][0],4)

        plt.text(0.95, 0.90, f'Type: {label_lst[int(func_info["Type"][0])]}      CV_WT: {cv_WT}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',horizontalalignment='right')
        plt.text(0.95, 0.85, f'CV: {cv}      PE: {pe}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',horizontalalignment='right')
        plt.text(0.95, 0.80, f'Period: {period}      Period Strength: {ps}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',horizontalalignment='right')
        
        # 设置坐标轴刻度标签的大小
        plt.tick_params(axis='x', direction='out',
                    labelsize=12, length=3.6)
        plt.tick_params(axis='y', direction='out',
                    labelsize=12, length=3.6)
        plt.savefig('/home/dell/xyf/azurefunctions-dataset2019/LazyProphet_exp/union_cvWT[2,5]_pe0.1/'+func+'.png')
        # plt.show()
        plt.clf()
        plt.close()
    gc.collect()    