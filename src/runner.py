import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from pdb import set_trace
from sklearn.impute import MissingIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from itertools import chain


def output_test_files(predict_alert_keys:list,probs:list):
    '''
        input: alert keys and its corresponding probabilities
    '''
    all_alert_keys = set(pd.read_csv("data/submission_sample.csv").alert_key)
    other_alert_keys = all_alert_keys - set(predict_alert_keys)
    lines = sorted(zip(predict_alert_keys, probs), key=lambda x: x[1], reverse=True)
    lines = "\n".join(chain(
        ["alert_key,probability"],
        map(lambda x: f"{int(x[0])},{x[1]}", lines),
        map(lambda x: f"{x},0.0", other_alert_keys),
        [""]
    ))
    with open("data/prediction.csv", 'w') as f:
        f.write(lines)
        

if __name__ == "__main__":
    
    data_path = Path("data")
    with open(data_path / 'train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    train_x, valid_x, train_y, valid_y = train_test_split(train_data[:,:-1], train_data[:,-1], test_size=0.2, random_state=2)
    
    RF = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=1000, class_weight={0: 1, 1:100})
    RF.fit(train_x, train_y)
    #RF_valid_y = RF.predict_proba(valid_x)[:,1]
    RF_valid_y = RF.predict(valid_x)
    RF_valid_cf_matrix = metrics.confusion_matrix(valid_y, RF_valid_y)
    print(RF_valid_cf_matrix)
    
    output_rank = enumerate(sorted(zip(RF_valid_y,valid_y), key=lambda x:x[0], reverse=True))
    output_rank = list(filter(lambda x:x[1][1]==1, output_rank))
    #print("\n", (len(output_rank) - 1) / output_rank[-2][0])
    
    with open(data_path / 'test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    test_keys, test_x = test_data[:,0], test_data[:,1:]

    pred = RF.predict_proba(test_x)[:,1]
    np.set_printoptions(precision=5)
    output_test_files(test_keys,pred)