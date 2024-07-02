from LazyProphet import LazyProphet as lp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle as pkl
import polars as pl
import pandas as pd
from collections import defaultdict
import sys
sys.path.append('/home/dell/xyf/azurefunctions-dataset2019/analysis')

PROCESSED_12DAYS_DIR = "/home/dell/xyf/azurefunctions-dataset2019/processed/12days/"
DATASET_LENGTH = 12
valid_split_DAY = 9
label_lst = ['Unknown','Warm', 'Regular', "Appro-regular", "Dense", "Successive", "Plused", "Possible", "Corr"]
CV_UPPER_THRESHOLD = 5
CV_LOWER_THRESHOLD = 0.1
PE_THRESHOLD = 0.2
PERIOD_STRENGTH_THRESHOLD = 0.5

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

def sliding_window_prediction(model, train_data, valid_data, local_window, predict_window):
    valid_size = len(valid_data)
    predictions = []
    extended_train_data = np.copy(train_data)
    
    for start in range(0, valid_size, predict_window):
        # Update training data by including part of the validation data
        window_data = np.concatenate((extended_train_data, valid_data[:start]))
        
        if len(window_data) > local_window:
            window_data = window_data[-local_window:]
        
        model.fit(window_data)
        pred = model.predict(predict_window).flatten()
        pred = list(map(lambda x: round(x) if x > 0 else 0, pred))
        predictions.extend(pred)
    
    return predictions


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

    metric_df = pl.read_csv(PROCESSED_12DAYS_DIR+"metric.csv")

    type_list = []
    for func, state in func_class.items():
        type_list.append((func, state.type))
            
    type_df = pl.DataFrame(type_list, schema=['Function', 'Type'])
    df = metric_df.join(type_df, on='Function')

    LOCAL_WINDOW = 60*24
    PREDICT_WINDOW = 60

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
    pred_func_account = {}
    func_ids = ['8e8a52ada035bc6cf4de91c8793add793325684eb57235f7cee6d8ded348d01e',
                    '1d8f4a33ef584c9396a70cb3c5e1f71e4d847e1d345c03fe6c41306af9d07099',
                    'e79fff2fdf4c23cd6c1e2e09e73bfb29ad15d05fea859eb1a11f603247dd7b2d',
                    'b3543c1d08ef66ff13e0df698e0e9615e4a497ace232c3715779151427fe15d5',
                    'e370dfc0407dec1f01bd6318b1da2433550d901d1410efc5be13d5125f81ab6d',
                    '69c500c02979aad7251a896443f0351f17fda907b2a725fa64dfa3dfaf033e18',
                    '5eadc93e1f21871e5b6f0dad42568cd7ac1686845442351762e37df3d843d78e',
                    '41d5874c93e42b71124b2cc6439e6f6fdb5920e67163eebef372593b878741f8',
                    '4a7dd1ef6093fd18d6476fd9426b4c7c99dbc7841c410504944ae18fad2640f2',
                    'f1de419dc75ea0f629deaf936e0b65934cbf2bc444ffd7b3116e3a19dd108f11',
                    '5b0d972219faa458acb5feb4f0789281be007552960ca6a68c062ef8a5f205c1',
                    '9680d492f74f5227698a669a4805fc8577d47fbaa7da95dacd6fcb2ffaf58d42',
                    'eb8de460611a65172fda55679873c739ce67592e478bf6b773e1919fa138eaf5',
                    '5336bda8faa5d11baac08b366930d5a0f89d37430f6df30028d75e7fb9724b3e',
                    '07cbff1ae225244f3c6bb9ed4bc1375fa762be6e097b3b62b862dc41ef3380ce',
                    '22072b261570f0452b5c8ae0cbe6e52c38ec68f19f1276535d87d02d7e085275',
                    'be58d43a3cd2c049eccfed0f645eff89d278fc1b45e7a08a9278c02e6214ec36',
                    '96b14a4f4e75557932044a3728918aa04102f9aa1dcd6e5ae8fd0e77b68b3dbf',
                    '0e205a2068ab2ffe80f5297bc77f2a9a60341394575a05a2114ec58f6926a1e1',
                    '3f028fd2cab9475007b3cff8fce6acbfb9c9e8f277afb98a5b6b86a77207bf7c']
    
    for LOCAL_WINDOW in range(60*6, 60*48+1, 60*6):
        for PREDICT_WINDOW in range(15, 60*2+1, 15):
            print('{}_{}'.format(LOCAL_WINDOW, PREDICT_WINDOW))
            with tqdm(total=len(func_ids)) as pbar:
                for func in func_ids:

                    lp_model = lp.LazyProphet(scale=True,
                                    seasonal_period=[24, 168],
                                    n_basis=8,
                                    fourier_order=10,
                                    ar=list(range(1, 97)),
                                    decay=.99,
                                    linear_trend=None,
                                    decay_average=False,
                                    boosting_params=boosting_params
                                    )
                    
                    arr = train_func_arrcount[func]
                    train_arr, valid_arr = arr[:1440 * valid_split_DAY], arr[1440 * valid_split_DAY:]
                    pred_result = sliding_window_prediction(lp_model, train_arr, valid_arr, LOCAL_WINDOW, PREDICT_WINDOW)
                    pred_func_account[func] = pred_result
                    pbar.update(1)

            with open('{}_{}.pkl'.format(LOCAL_WINDOW, PREDICT_WINDOW), 'wb') as f:
                pkl.dump(pred_func_account, f)