import os
import yaml
import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import Counter

from pdb import set_trace

def get_downstream_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', '--neg', type=int, required=True)
    parser.add_argument('-p', '--pos', type=int, required=True)
    parser.add_argument('-d', '--day', type=int, required=True)
    parser.add_argument( '--test', action='store_true')
    parser.add_argument( '--train', action='store_true')
    args = parser.parse_args()
    if args.neg > 243: raise ValueError
    if args.pos > 20000: raise ValueError

    return args

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
    
if __name__ == "__main__":
    args = get_downstream_args()

    data_path = Path("data")
    tables = load_tables(data_path)
    # #train
    if args.train:
        sas_alertKeys, non_sas_alertKeys = find_sas_nonsas_key(args.pos,args.neg)
        os.makedirs(data_path/ 'analyze', exist_ok=True)
        sas = alerkey_find_info(sas_alertKeys, args.day, tables, data_path/ 'analyze' /'sas_data.csv')
        nonsas = alerkey_find_info(non_sas_alertKeys, args.day, tables, data_path/ 'analyze' /'non_sas.csv')
        data = form_dataset(sas, nonsas)
        with open(data_path / "train_data.pkl", 'wb') as f:
            pickle.dump(data, f)

    #test
    if args.test:
        test_keys = pd.read_csv(data_path / "public_x_alert_date.csv")['alert_key'].to_numpy()
        test_data = alerkey_find_info(test_keys, args.day, tables, data_path/ 'analyze' /'test.csv')
        test_data = form_testset(test_data)
        with open(data_path / "test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)
    