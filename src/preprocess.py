import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
from pdb import set_trace

class IndexCounter:
    def __init__(self, start_idx=0):
        self.idx = start_idx - 1
        
    def __iter__(self):
        while True:
            self.idx += 1
            yield self.idx
    def __next__(self):
        self.idx += 1
        return self.idx

def load_tables(data_path):
    tables = {}
    tables["ccba"] = pd.read_csv(data_path / "public_train_x_ccba_full_hashed.csv")
    tables["cdtx"] = pd.read_csv(data_path / "public_train_x_cdtx0001_full_hashed.csv")
    tables["custinfo"] = pd.read_csv(data_path / "public_train_x_custinfo_full_hashed.csv")
    tables["dp"] = pd.read_csv(data_path / "public_train_x_dp_full_hashed.csv")
    tables["remit"] = pd.read_csv(data_path / "public_train_x_remit1_full_hashed.csv")
    
    public_x = pd.read_csv(data_path / "public_x_alert_date.csv")
    train_x = pd.read_csv(data_path / "train_x_alert_date.csv")
    # merge date to custinfo
    tables["custinfo"] = tables["custinfo"].merge(
        pd.concat((train_x, public_x), ignore_index=True),
        left_on="alert_key", right_on="alert_key"
    ).sort_values(by="alert_key")
  
    return tables

def alerkey_find_info(alertkeys, retri_time, tables, path): #use alert key to find last retri_time ccba, cdtx,...info
    info = []
    for i, alertkey in enumerate(alertkeys):
        cust_info = tables['custinfo'].loc[tables['custinfo']['alert_key'] == alertkey]
        cust_id, date = cust_info['cust_id'].item(), cust_info['date'].to_numpy().item()
        start_date = (date - retri_time) if (date - retri_time) > 0 else 0
        ccba = tables['ccba'].loc[(tables['ccba']['cust_id'] == cust_id)]
        if len(ccba) == 0: ccba = pd.DataFrame(np.zeros((13,len(ccba.columns))),columns=ccba.columns)
        elif len(ccba) < 13: 
            repeat_last_row = np.tile(ccba.to_numpy()[-1,:], 13-len(ccba)).reshape(13-len(ccba),-1)
            ccba = pd.DataFrame(np.append(ccba.to_numpy(), repeat_last_row,axis=0),columns=ccba.columns)
        
        if date <= 30: ccba = ccba.iloc[[0]]
        elif date > 30 and date < 390: 
            ccba = ccba.iloc[[int(date/30)-1,int(date/30)]]
        else: ccba = ccba.iloc[[11,12]]
        cdtx = tables['cdtx'].loc[(tables['cdtx']['cust_id'] == cust_id) & ((start_date<tables['cdtx']['date'].to_numpy()) & (tables['cdtx']['date'].to_numpy()<=date))]
        dp = tables['dp'].loc[(tables['dp']['cust_id'] == cust_id) & ((start_date<tables['dp']['tx_date'].to_numpy()) & (tables['dp']['tx_date'].to_numpy()<=date))]
        remit = tables['remit'].loc[(tables['remit']['cust_id'] == cust_id) & ((start_date<tables['remit']['trans_date'].to_numpy()) & (tables['remit']['trans_date'].to_numpy()<=date))]
        
        if i == 0:
            info = processing_data(cust_info, ccba, cdtx, dp, remit)
        else:       
            append = processing_data(cust_info, ccba, cdtx, dp, remit)
            info = pd.concat([info,append],ignore_index=True)

    info.to_csv(path, index=False, header=True)
     
    return info

def processing_data(custinfo, ccba, cdtx, dp, remit):
    cust_id =  custinfo['cust_id'].to_numpy()
    alert_key = custinfo['alert_key'].to_numpy()
    risk_rank = custinfo['risk_rank'].to_numpy()
    occupation_code = custinfo['occupation_code'].to_numpy()
    total_asset = custinfo['total_asset'].to_numpy()
    AGE = custinfo['AGE'].to_numpy()
    if np.isnan(occupation_code): occupation_code=0
    
    lupay = ccba['lupay'].to_numpy().mean()
    cycam = ccba['cycam'].to_numpy().mean()
    usgam = ccba['usgam'].to_numpy().mean()
    clamt = ccba['clamt'].to_numpy().mean()
    csamt = ccba['csamt'].to_numpy().mean()
    inamt = ccba['inamt'].to_numpy().mean()
    cucsm = ccba['cucsm'].to_numpy().mean()
    cucah = ccba['cucah'].to_numpy().mean() 
    del ccba["byymm"]
    del ccba["cust_id"]
    
    country = np.where(cdtx['country'].to_numpy()==130,0,1).mean()
    cur_type = np.where(cdtx['cur_type'].to_numpy()==47,0,1).mean()
    amt = cdtx['amt'].to_numpy().sum()
    cdtx_freq = len(cdtx)
    if np.isnan(country): country=-1
    if np.isnan(cur_type): cur_type=-1
    if np.isnan(amt): amt=0
    del cdtx["cust_id"]
    del cdtx['date'] 
    
    tx_amt = (dp['tx_amt'] * dp['exchg_rate']).to_numpy().sum()
    debit_credit = np.where(dp['debit_credit'].to_numpy()=='DB',0,1).mean()
    tx_type = dp['tx_type'].to_numpy()
    if len(tx_type) == 0: tx_type = np.zeros(3)
    else:   tx_type = np.bincount(tx_type,minlength=4)[1:]/np.bincount(tx_type,minlength=4).sum()
        
    cross_bank = dp['cross_bank'].to_numpy().mean()
    ATM = dp['ATM'].to_numpy().mean()
    dp_freq = len(dp)
    if np.isnan(tx_amt): tx_amt=0
    if np.isnan(debit_credit): debit_credit=-1
    if np.isnan(cross_bank): cross_bank=-1
    if np.isnan(ATM): ATM=-1
    del dp["cust_id"]
    del dp['tx_date']
    del dp['tx_time']
    del dp['exchg_rate']
    del dp['info_asset_code']
    del dp['fiscTxId']
    del dp['txbranch']
    
    trans_no = remit['trans_no'].to_numpy()  
    if len(trans_no) == 0: trans_no = np.zeros(5)
    else:   trans_no = np.bincount(trans_no,minlength=5)/np.bincount(trans_no,minlength=5).sum()
    
    trade_amount_usd = (remit['trade_amount_usd'].to_numpy()).sum()
    remit_freq = len(remit)
    if np.isnan(trade_amount_usd): trade_amount_usd=0
    del remit['cust_id']
    del remit['trans_date']
    
    info = {'cust_id':cust_id,'alert_key':alert_key,'risk_rank':risk_rank,'occupation_code':occupation_code
            ,'total_asset':total_asset,'AGE':AGE,'lupay':lupay, 'cycam':cycam,'usgam':usgam,'clamt':clamt
            ,'csamt':csamt,'inamt':inamt,'cucsm':cucsm,'cucah':cucah,'country':country, 'cur_type':cur_type
            ,'amt':amt,'cdtx_freq':cdtx_freq,'tx_amt':tx_amt,'debit_credit':debit_credit,'tx_type1':tx_type[0]
            ,'tx_type2':tx_type[1],'tx_type3':tx_type[2],'cross_bank':cross_bank,'ATM':ATM,'dp_freq':dp_freq
            ,'trans_no0':trans_no[0],'trans_no1':trans_no[1],'trans_no2':trans_no[2],'trans_no3':trans_no[3]
            ,'trans_no4':trans_no[4],'trade_amount_usd':trade_amount_usd,'remit_freq':remit_freq}

    data = pd.DataFrame.from_dict(info)

    if data.isnull().values.any(): 
        print(f"{np.argwhere(data.isnull().values)} is nan")
        raise ValueError 
    
    return data
    
def find_sas_nonsas_key(sas_num, non_sas_num):
    if sas_num > 234: raise ValueError
    train_y = pd.read_csv(data_path / "train_y_answer.csv")
    sas_alertkey = train_y.loc[(train_y['sar_flag']==1)]['alert_key'].to_numpy()
    sas_alertkey = np.random.choice(sas_alertkey,size=sas_num,replace=False)
    non_sas_alertkey = train_y.loc[(train_y['sar_flag']==0)]['alert_key'].to_numpy()
    non_sas_alertkey = np.random.choice(non_sas_alertkey,size=non_sas_num,replace=False)
    
    return sas_alertkey, non_sas_alertkey

def form_dataset(sas:pd, nonsas:pd):
    # return np dataset with [-1] = label
    sas_label = np.ones(len(sas))
    sas['label'] = sas_label
    nonsas_label = np.zeros(len(nonsas))
    nonsas['label'] = nonsas_label
    data = pd.concat([sas, nonsas],ignore_index=True)
    data.drop(['cust_id','alert_key'],axis=1,inplace=True)
    
    return data.to_numpy()
    
def form_testset(data:pd):
    data.drop(['cust_id'],axis=1,inplace=True)
    return data.to_numpy()

   
# def data_preprocess(tables, data_config, data_path):
#     mapping = {}
#     for table_name, table in tables.items():
#         idx_counter = IndexCounter()
#         for col_name in table:
#             cfg = data_config[table_name][col_name]
#             if cfg['type'] == "date":
#                 table[col_name] = table[col_name] / 365
#             elif cfg['type'] == "categorical":
#                 series = table[col_name].fillna(-1)
#                 labels = sorted(series.unique())
#                 mapping[col_name] = dict(zip(labels, idx_counter))
#                 table[col_name] = series.apply(lambda x: mapping[col_name][x])
#             elif cfg['type'] == "numerical":
#                 table[col_name] = (table[col_name] - table[col_name].mean()) / table[col_name].std()
#             elif cfg['type'] == "label":
#                 pass
#             else:
#                 raise NotImplementedError
#         set_trace()
#         tables[table_name] = table.groupby("cust_id")
    
#     # save the mapping between original label and the index for embedding
#     if (data_path / "mapping.pkl").exists():
#         with open(data_path / "mapping.pkl", 'rb') as f:
#             assert mapping == pickle.load(f), "mapping is different"
#     else:
#         with open(data_path / "mapping.pkl", 'wb') as f:
#             pickle.dump(mapping, f)
        
#     return tables
        
# def concat_data(grouped_tables, data_config):
#     data = {key: {} for key in grouped_tables["custinfo"].groups.keys()}
#     for table_name, table in grouped_tables.items():
#         data_type_counter = Counter(v['type'] for v in data_config[table_name].values())
#         for cust_id, df in table:
#             num = len(df)
#             data[cust_id][table_name] = {}
#             cat_idx_counter = IndexCounter()
#             num_idx_counter = IndexCounter()
#             categorical = np.empty((num, data_type_counter["categorical"]), dtype=np.int64)
#             numerical = np.empty((num, data_type_counter["numerical"]), dtype=np.float32)
#             for col_name in df:
#                 cfg = data_config[table_name][col_name]
#                 if cfg['type'] == "date":
#                     date = np.array(df[col_name], dtype=np.float32)
#                 elif cfg['type'] == "categorical":
#                     categorical[:, next(cat_idx_counter)] = df[col_name].to_numpy(dtype=np.int64)
#                 elif cfg['type'] == "numerical":
#                     numerical[:, next(num_idx_counter)] = df[col_name].to_numpy(dtype=np.float32)
#                 elif cfg['type'] == "label":
#                     data[cust_id][table_name][col_name] = df[col_name].tolist()
#                 else:
#                     raise NotImplementedError

#             data[cust_id][table_name]["date"] = date
#             data[cust_id][table_name]["cat"] = categorical
#             data[cust_id][table_name]["num"] = numerical
    
#     for cust_id, tables in list(data.items()):
#         event_index = []
#         for table_name, table in tables.items():
#             event_index.extend(((table_name, i, date) for i, date in enumerate(table["date"])))
#         event_index.sort(key=lambda x: x[-1])
#         data[cust_id]["event_index"] = event_index
    
#     return data
    
    
if __name__ == "__main__":
    data_path = Path("data")
    with open(data_path / "data_config.yaml", 'r') as f:
        data_config = yaml.load(f, yaml.Loader)

    tables = load_tables(data_path)
    # grouped_tables = data_preprocess(tables, data_config, data_path)
    # data = concat_data(grouped_tables, data_config)
   
    # #train
    # sas_alertKeys, non_sas_alertKeys = find_sas_nonsas_key(234,20000)
    # sas = alerkey_find_info(sas_alertKeys, 30, tables, data_path/ 'analyze' /'sas_data.csv')
    # nonsas = alerkey_find_info(non_sas_alertKeys, 30, tables, data_path/ 'analyze' /'non_sas.csv')
    # data = form_dataset(sas, nonsas)
    # with open(data_path / "data.pkl", 'wb') as f:
    #     pickle.dump(data, f)

    #test
    test_keys = pd.read_csv(data_path / "public_x_alert_date.csv")['alert_key'].to_numpy()
    test_data = alerkey_find_info(test_keys, 30, tables, data_path/ 'analyze' /'test.csv')
    test_data = form_testset(test_data)
    with open(data_path / "test_data.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    
    

    """
    The final data will be something like:
    {
        cust_id: {
            table_name: {
                date: np.ndarray (shape: (num, ))
                cat: np.ndarray(shape: (num, # of categorical features))
                num: np.ndarray(shape: (num, # of numerical features))
            }
            ...
            "event_index": [(table_name, index), ...]
        }
        ...
    }
    """
        