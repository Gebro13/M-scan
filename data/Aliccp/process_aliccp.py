import pandas as pd
import numpy as np
import pickle as pkl
import os
import random
from tqdm import tqdm
import time
import copy
from collections import Counter
import matplotlib.pyplot as plt
import datetime
import scipy.stats
import sys
import gc

DATA_DIR = ''

 
def subject(file1_in,file2_in):
    data = pd.read_csv(file1_in)
    print(sum(data['click']))
    print(data['user_id'].unique().size)
    data_ts = pd.read_csv(file2_in)
    print(sum(data_ts['click']))
    # train_raw = pd.read_csv('feateng_data/train_raw.csv',names = ['click','domain_id','item_id','206','207','216','user_id','121','122','124','125','126','127','128','129'])
    # test_raw = pd.read_csv('feateng_data/test_raw.csv',names = ['click','domain_id','item_id','206','207','216','user_id','121','122','124','125','126','127','128','129'])
    # train_users = train_raw['user_id'].unique()
    # test_users = test_raw['user_id'].unique()
    # print(len(np.intersect1d(train_users,test_users)))
    # print(len(train_users))
    # print(len(test_users))
    
    print(data_ts['user_id'].unique().size)
def process_data(file1_in,file2_in,file_out):
    user_f = open(file1_in,'r')
    except_list = ['109_14','110_14','127_14','150_14']
    user_feat_dict = {}
    cnt= 0
    for line in tqdm(user_f):
        user_id = line.strip().split(',')[0]
        
        user_feat_dict[user_id] = {}
        features = line.strip().split(',')[2].split('\x01')
        for feat in features:
            field_id = feat.split('\x02')[0]
            if field_id in except_list:
                continue
            feat_id = feat.split('\x02')[1].split('\x03')[0]
            if field_id in user_feat_dict[user_id]:
                user_feat_dict[user_id][field_id] = str(user_feat_dict[user_id]) + '#' +str(feat_id)
            else:
                user_feat_dict[user_id][field_id] = str(feat_id)
       
    user_f.close()
    print('cnt of users: ',len(user_feat_dict))
    sample_f = open(file2_in,'r')
    merge_f = open(file_out,'w')
    
    #merge_f.write(','.join(['click','domain_id','item_id','']))
    #rm the combination features
    feat_exp_list = ['508','509','702','853']
    for line in tqdm(sample_f):
        line_split = line.strip().split(',')
        click = line_split[1]
        user_id = line_split[3]
        features = line_split[5].split('\x01')
        feat_map = {}
        for feat in features:
            field_id = feat.split('\x02')[0]
            feat_id = feat.split('\x02')[1].split('\x03')[0]
            if field_id in feat_exp_list:
                continue
            if field_id in feat_map:
                if type(feat_map[field_id]) == list:
                    feat_map[field_id].append(feat_id)
                else:
                    feat_map[field_id] = [feat_map[field_id],feat_id]
            else:
                feat_map[field_id] = feat_id
        # if type(feat_map.get('210','')) != list:
        #     feat_map['210'] = [feat_map['210']]
        # item_feats = [click,feat_map.get('301',''),feat_map.get('205',''),feat_map.get('206',''),feat_map.get('207',''),'#'.join(feat_map['210']),feat_map.get('216','')]
        item_feats = [click,feat_map.get('301','-1'),feat_map.get('205','-2'),feat_map.get('206','-3'),feat_map.get('207','-4'),feat_map.get('216','-5')]
        user_feats = [user_feat_dict[user_id].get('101','-6'),user_feat_dict[user_id].get('121','-7'),user_feat_dict[user_id].get('122','-8'),user_feat_dict[user_id].get('124','-9'),user_feat_dict[user_id].get('125','-10'),user_feat_dict[user_id].get('126','-11'),user_feat_dict[user_id].get('127','-12'),user_feat_dict[user_id].get('128','-13'),user_feat_dict[user_id].get('129','-14')]
        # user_feats = [user_id,user_feat_dict[user_id].get('121','-7'),user_feat_dict[user_id].get('122','-8'),user_feat_dict[user_id].get('124','-9'),user_feat_dict[user_id].get('125','-10'),user_feat_dict[user_id].get('126','-11'),user_feat_dict[user_id].get('127','-12'),user_feat_dict[user_id].get('128','-13'),user_feat_dict[user_id].get('129','-14')]
        new_line = ','.join(item_feats + user_feats) + '\n'
        merge_f.write(new_line)
        # break
       

def remap_all_data(train_file_in,train_file_out,test_file_in,test_file_out):

    print('start remapping')
    train_data = pd.read_csv(train_file_in,names = ['click','domain_id','item_id','206','207','216','user_id','121','122','124','125','126','127','128','129'])
    test_data = pd.read_csv(test_file_in,names = ['click','domain_id','item_id','206','207','216','user_id','121','122','124','125','126','127','128','129'])
    data_all = pd.concat([train_data,test_data])
    train_len = len(train_data)
    # for col in data_all:
    #     print('col names:{} size:{}'.format(col,data_all[col].unique().size))
    print(data_all)

    remap_num = 1
    remapped_dict = {}
    data = data_all[['user_id','121','122','124','125','126','127','128','129','item_id','206','207','216','domain_id','click']]
    data = data.to_numpy()
    for j in tqdm(range(len(data[0])-1)):
        for i in range(len(data)):
            if data[i][j] not in remapped_dict:
                remapped_dict[data[i][j]] = remap_num
                data[i][j] = remap_num
                remap_num+=1
            else:
                data[i][j] = remapped_dict[data[i][j]]

    remap_train = data[:train_len]
    remap_test = data[train_len:]
    print(1)
    remap_train = pd.DataFrame(remap_train,columns = ['user_id','121','122','124','125','126','127','128','129','item_id','206','207','216','domain_id','click'])
    remap_test = pd.DataFrame(remap_test,columns = ['user_id','121','122','124','125','126','127','128','129','item_id','206','207','216','domain_id','click'])
    print(remap_train)

    remap_train.to_csv(train_file_out,index=False)
    print(2)
    remap_test.to_csv(test_file_out,index=False)
    print(remap_train)
    print('remap finished')

def generate_user_item_dict(train_, test_):
    train_data = pd.read_csv(train_)
    test_data = pd.read_csv(test_)
    # train_data = pd.read_csv(train_,names = ['click','domain_id','item_id','206','207','216','user_id','121','122','124','125','126','127','128','129'])
    # test_data = pd.read_csv(test_,names = ['click','domain_id','item_id','206','207','216','user_id','121','122','124','125','126','127','128','129'])
    
    #用test_data里的item_dict
    data_all = pd.concat([test_data,train_data])
    data_all = data_all[['user_id','121','122','124','125','126','127','128','129','item_id','206','207','216','domain_id','click']]
    data_all = data_all.to_numpy()
    item_dict = {}
    user_dict = {}
    error_cnt = 0
    for i in tqdm(range(len(data_all))):
        row = data_all[i]
        user = row[0]
        user_features = list(row[1:9])
        if user in user_dict and user_features != user_dict[user]:
            error_cnt +=1
            print('error')
        else:
            user_dict[user] = user_features
        item = row[9]
        item_features = list(row[10:13])
        
        if item in item_dict and item_features != item_dict[item]:
            print('Error')
        else:
            item_dict[item] = item_features
    print(error_cnt)
    with open('feateng_data/user_dict.pkl','wb') as f:
        pkl.dump(user_dict,f)
    with open('feateng_data/item_dict.pkl','wb') as f:
        pkl.dump(item_dict,f)
    

def generate_user_behavs_dict(train_,test_,hist_file,train_file,test_file):
    print(1)

    train_data_old = pd.read_csv(train_)
    
    test_data_old = pd.read_csv(test_) 
    
    
    user_behavs_dict = {}

    user_theme_behavs_dict = {}
    fake_time = 0

    for row in tqdm(train_data_old.itertuples(index=False)):
        fake_time+=1
        user, item, theme,click = row[0], row[9], row[13],row[14]
        
        #user_behavs_dict 
        if user in user_behavs_dict:
            user_behavs_dict[user].append((item,click,fake_time))
        else:
            user_behavs_dict[user] = [(item,click,fake_time)]
        #user_theme_behavs_dict
        if user in user_theme_behavs_dict:
            if theme in user_theme_behavs_dict[user]:
                user_theme_behavs_dict[user][theme].append((item,click,fake_time))
            else:
                user_theme_behavs_dict[user][theme] = [(item,click,fake_time)]
        else:
            user_theme_behavs_dict[user] = {theme:[(item,click,fake_time)]}
    

    user_theme_behavs_test_dict = {}

    user_behavs_test_dict = {}
    for row in tqdm(test_data_old.itertuples(index=False)):
        fake_time +=1
        user, item, theme, click = row[0], row[9], row[13],row[14]
        
        #user_behavs_dict 
        if user in user_behavs_test_dict:
            user_behavs_test_dict[user].append((item,click,fake_time))
        else:
            user_behavs_test_dict[user] = [(item,click,fake_time)]
        #user_theme_behavs_dict
        if user in user_theme_behavs_test_dict:
            if theme in user_theme_behavs_test_dict[user]:
                user_theme_behavs_test_dict[user][theme].append((item,click,fake_time))
            else:
                user_theme_behavs_test_dict[user][theme] = [(item,click,fake_time)]
        else:
            user_theme_behavs_test_dict[user] = {theme:[(item,click,fake_time)]}
    
    print(fake_time)
    with open('feateng_data/user_behavs_dict.pkl','wb') as f:
        pkl.dump(user_behavs_dict,f)
    with open('feateng_data/user_theme_behavs_dict.pkl','wb') as f:
        pkl.dump(user_theme_behavs_dict,f)

    with open('feateng_data/user_behavs_test_dict.pkl','wb') as f:
        pkl.dump(user_behavs_test_dict,f)
    with open('feateng_data/user_theme_behavs_test_dict.pkl','wb') as f:
        pkl.dump(user_theme_behavs_test_dict,f)

    
    with open('feateng_data/user_behavs_dict.pkl','rb') as f:
        user_behavs_dict = pkl.load(f)
    with open('feateng_data/user_theme_behavs_dict.pkl','rb') as f:
        user_theme_behavs_dict = pkl.load(f)
        
    with open('feateng_data/user_behavs_test_dict.pkl','rb') as f:
        user_behavs_test_dict= pkl.load(f)
    with open('feateng_data/user_theme_behavs_test_dict.pkl','rb') as f:
        user_theme_behavs_test_dict= pkl.load(f)

    user_behavs_all_dict = user_behavs_dict.copy()
    user_theme_behavs_all_dict = user_theme_behavs_dict.copy()
    for uid, behavs in tqdm(user_behavs_test_dict.items()):
        if uid in user_behavs_all_dict:
            user_behavs_all_dict[uid] += behavs
        else:
            user_behavs_all_dict[uid] = behavs
    
    for uid in tqdm(user_theme_behavs_test_dict.keys()):
        for theme, behavs in user_theme_behavs_test_dict[uid].items():
            if uid in user_theme_behavs_all_dict:
                if theme in user_theme_behavs_all_dict[uid]:
                    user_theme_behavs_all_dict[uid][theme]+=behavs
                else:
                    user_theme_behavs_all_dict[uid][theme] = behavs
            else:
                user_theme_behavs_all_dict[uid] = {theme:behavs}
    
    with open('feateng_data/user_behavs_all_dict.pkl','wb') as f:
        pkl.dump(user_behavs_all_dict,f)
    with open('feateng_data/user_theme_behavs_all_dict.pkl','wb') as f:
        pkl.dump(user_theme_behavs_all_dict,f)

    

def dataset_statistics():


    t1 = time.time()
    with open('feateng_data/user_behavs_dict.pkl','rb') as f:
        user_behavs_dict = pkl.load(f)
    with open('feateng_data/user_theme_behavs_dict.pkl','rb') as f:
        user_theme_behavs_dict = pkl.load(f)
        
    with open('feateng_data/user_behavs_test_dict.pkl','rb') as f:
        user_behavs_test_dict= pkl.load(f)
    with open('feateng_data/user_theme_behavs_test_dict.pkl','rb') as f:
        user_theme_behavs_test_dict= pkl.load(f)


    lengths =[]
    cnt=0
    user_cnt=0
    for uid in user_behavs_test_dict.keys():
        if uid in user_behavs_dict and len(user_behavs_dict[uid])>10:
            cnt+=1

        if uid in user_behavs_dict:
            user_cnt+=1
            lengths.append(len(user_behavs_dict[uid]))

    print(cnt)
    print(user_cnt)
    print(len(user_behavs_dict.keys()))

    counter = Counter(lengths)
    sorted_items = sorted(counter.items(), key=lambda x: x[0])
    x = []
    y = []
    for items in sorted_items:
        x.append(items[0])
        y.append(items[1])
    import matplotlib.pyplot as plt

    plt.plot(x,y)
    # 添加标签和标题
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.xlim(left = 0,right=100)

    # 显示图形
    plt.show()
    plt.savefig('1.png', dpi=300, bbox_inches='tight')
    lengths = []
    cold_cnt=0
    cnt=0
    multi_theme_user_cnt = 0
    for uid in tqdm(user_theme_behavs_test_dict.keys()):
        if len(user_theme_behavs_test_dict[uid].keys()) >1:
            multi_theme_user_cnt +=1
        for theme in user_theme_behavs_test_dict[uid].keys():
            if len(user_theme_behavs_test_dict[uid][theme]) <20:
                cold_cnt+=1
            cnt+=1
            lengths.append(len(user_theme_behavs_test_dict[uid][theme]))
    print(cnt)
    print(cold_cnt)
    print(multi_theme_user_cnt)
    
    counter = Counter(lengths)
    sorted_items = sorted(counter.items(), key=lambda x: x[0])
    x = []
    y = []
    for items in sorted_items:
        x.append(items[0])
        y.append(items[1])
    import matplotlib.pyplot as plt

    plt.plot(x,y)
    # 添加标签和标题
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.xlim(left = 0,right=100)

    # 显示图形
    plt.show()
    
    plt.savefig('4.png', dpi=300, bbox_inches='tight')

    print(time.time()-t1)

def resplit_hist_train_test(train_,test_,hist_file,train_file,test_file):

    print('Train_test respliting...')
    with open('feateng_data/user_behavs_all_dict.pkl','rb') as f:
        user_behavs_all_dict = pkl.load(f)
    with open('feateng_data/user_theme_behavs_all_dict.pkl','rb') as f:
        user_theme_behavs_all_dict=pkl.load(f)

    print(len(user_theme_behavs_all_dict.keys()))
    
    def padding(x,padded_length):
        length = len(x)
        if length>=padded_length:
            return x[-padded_length:]
        else:
            return np.pad(x, ((0, padded_length-length), (0, 0)), mode='constant')



    x_train = []
    y_train = []
    x_theme_hist_train = []
    x_hist_train = []
    x_test = []
    y_test = []
    x_theme_hist_test = []
    x_hist_test = []
    cnt=0
    lengths = []
    
    for uid in tqdm(user_theme_behavs_all_dict.keys()):
        if len(user_theme_behavs_all_dict[uid].keys())<=1:
            continue
        cnt+=1
        for theme,theme_behavs in user_theme_behavs_all_dict[uid].items():
           
            
            if len(theme_behavs)>21 and len(theme_behavs)<5000:
        
                theme_behavs =np.array(theme_behavs)
                all_behavs = np.array(user_behavs_all_dict[uid])
                # print(len(all_behavs))
                avail_behavs = theme_behavs[40:]
                length = len(avail_behavs)
                for i in range(int(0.6*length)):
                    behav = avail_behavs[i]
                    # 同theme的behavior
                    item,click,time = behav
                    theme_behavs_chosen = theme_behavs[theme_behavs[:,2]<time]
                    
                    theme_behavs_chosen = theme_behavs_chosen[theme_behavs_chosen[:,1]==1]
                    
                    # print(len(theme_behavs_chosen))
                    
                    lengths.append(len(theme_behavs_chosen[-20:]))
                    theme_behavs_chosen = padding(theme_behavs_chosen,20)
                    # print(theme_behavs_chosen)
                    # 不同theme的behavior
                    
                    # lengths.append(len(theme_behavs_chosen))
                    behavs_chosen = all_behavs[all_behavs[:,2]<time]
                    behavs_chosen = padding(behavs_chosen,40)
            
                    x_train.append([uid,item,theme])
                    y_train.append(click)
            
                    x_theme_hist_train.append(theme_behavs_chosen)
                    x_hist_train.append(behavs_chosen)   
                   
                for i in range(int(0.6*length),int(length)):
                    behav = avail_behavs[i]
                    # 同theme的behavior
                    item,click,time = behav
                    theme_behavs_chosen = theme_behavs[theme_behavs[:,2]<time]
                    
                    theme_behavs_chosen = theme_behavs_chosen[theme_behavs_chosen[:,1]==1]
                    
                    theme_behavs_chosen = padding(theme_behavs_chosen,20)
                     # 不同theme的behavior
                    
                    # lengths.append(len(theme_behavs_chosen))
                    behavs_chosen = all_behavs[all_behavs[:,2]<time]
                    behavs_chosen = padding(behavs_chosen,40)
                    x_test.append([uid,item,theme])
                    y_test.append(click)
                    x_theme_hist_test.append(theme_behavs_chosen)
                    x_hist_test.append(behavs_chosen)   
            
    print(np.mean(lengths))
    print(cnt)
    print(len(x_train))
    print(len(x_test))
    x_train,y_train,x_theme_hist_train,x_hist_train = np.array(x_train),np.array(y_train),np.array(x_theme_hist_train),np.array(x_hist_train)
    x_test,y_test,x_theme_hist_test,x_hist_test = np.array(x_test),np.array(y_test),np.array(x_theme_hist_test),np.array(x_hist_test)

    with open('feateng_data/train_set_.pkl','wb') as f:
        pkl.dump([x_train,y_train,x_theme_hist_train,x_hist_train],f)
    
    with open('feateng_data/test_set_.pkl','wb') as f:
        pkl.dump([x_test,y_test,x_theme_hist_test,x_hist_test],f)

def generate_full_train_and_test_set(train_set_,test_set_,train_set,test_set):
    
    with open('feateng_data/user_dict.pkl','rb') as f:
        user_dict = pkl.load(f)
    with open('feateng_data/item_dict.pkl','rb') as f:
        item_dict = pkl.load(f)
    with open(train_set_,'rb') as f:
        a=pkl.load(f)
        print(sys.getsizeof(a))
        x_train,y_train,x_theme_hist_train,x_hist_train = a
    with open(test_set_,'rb') as f:
        x_test,y_test,x_theme_hist_test,x_hist_test = pkl.load(f)
    
    print(x_train),print(y_train),print(x_theme_hist_train),print(x_hist_train)

    def generate_full(x,x_theme_hist,x_hist):
        x_full,x_theme_hist_full,x_hist_full = [],[],[]
        for row in tqdm(x):
            uid,item,domain = row
            x_full.append([uid]+user_dict[uid]+[item]+item_dict[item]+[domain])
        for theme_behavs in tqdm(x_theme_hist):
            theme_behavs_full = []
            for theme_behav in theme_behavs:
                item,click,time = theme_behav
                if item in item_dict:
                    theme_behavs_full.append([item]+item_dict[item]+[click])
                else:
                    theme_behavs_full.append([0]*5)
            x_theme_hist_full.append(theme_behavs_full)
        for behavs in tqdm(x_hist):
            behavs_full = []
            for behav in behavs:
                item,click,time = behav
                behavs_full.append([item]+item_dict[item]+[click])
            x_hist_full.append(behavs_full)
        return [np.array(_) for _ in [x_full,x_theme_hist_full,x_hist_full]]
    x_train,x_theme_hist_train,x_hist_train = generate_full(x_train,x_theme_hist_train,x_hist_train)
    
    print(x_train)
    print(x_theme_hist_train,x_hist_train)
    x_test,x_theme_hist_test,x_hist_test = generate_full(x_test,x_theme_hist_test,x_hist_test)
    

    with open(train_set,'wb') as f:
        pkl.dump([x_train,y_train,x_theme_hist_train,x_hist_train],f)
    
    with open(test_set,'wb') as f:
        pkl.dump([x_test,y_test,x_theme_hist_test,x_hist_test],f)
    
    


def generate_train_and_test_set(flag,file_in,file_out,file_out2):
    data_all = pd.read_csv(file_in)
    data_all = data_all.to_numpy()
    x = data_all[:,:-1]
    y = data_all[:,-1]
    print(x.shape)
    print(y.shape)
    with open(file_out,'wb') as f:
        pkl.dump([x,y],f,protocol=4)
    if flag == 'test':
        with open(file_out2,'wb') as f:
            pkl.dump([x[:2000000],y[:2000000]],f,protocol=4)
    
def split_train_domain(train_file_in,train_domain_files):
    train_data = pd.read_csv(train_file_in)
    domains = [5732617,5732618,5732619]
    for domain,file_out in tqdm(zip(domains,train_domain_files)):
        train_domain = train_data[train_data['domain_id'] == domain].to_numpy()
        x = train_domain[:,:-1]
        y = train_domain[:,-1]
        print(x.shape)
        with open(file_out,'wb') as f:
            pkl.dump([x,y],f,protocol=4)

def split_test_domain(test_file_in,test_domain_files):
    with open(test_file_in,'rb') as f:
        x_test,y_test,x_theme_hist,x_hist = pkl.load(f)
    print(x_test)
    domains = [5732620,5732618,5732619]
    for domain,file_out in tqdm(zip(domains,test_domain_files)):
        domain_idx = x_test[:,-1] == domain
        print(sum(domain_idx))
        x_test_domain,y_test_domain,x_theme_hist_domain, x_hist_domain = x_test[domain_idx],y_test[domain_idx],x_theme_hist[domain_idx],x_hist[domain_idx]

        with open(file_out,'wb') as f:
            pkl.dump([x_test_domain,y_test_domain,x_theme_hist_domain, x_hist_domain],f,protocol=4)
        



if __name__ == '__main__':
    # process_data('raw_data/common_features_train.csv','raw_data/sample_skeleton_train.csv','feateng_data/train_raw.csv')
    # process_data('raw_data/common_features_test.csv','raw_data/sample_skeleton_test.csv','feateng_data/test_raw.csv')
    # subject('feateng_data/train.csv','feateng_data/test.csv')
    # read_all_data('raw_data/sample_skeleton_train.csv','raw_data/common_features_train.csv')
    # remap_all_data('feateng_data/train_raw.csv','feateng_data/train_.csv','feateng_data/test_raw.csv','feateng_data/test_.csv')
    # generate_user_item_dict('feateng_data/train_.csv','feateng_data/test_.csv')
    # generate_user_behavs_dict('feateng_data/train_.csv','feateng_data/test_.csv','feateng_data/hist.csv','feateng_data/train.csv','feateng_data/test.csv')
    # dataset_statistics()
    
    # print(len(x_hist_train))
    # print(x_hist_train.shape)
    # resplit_hist_train_test('feateng_data/train_.csv','feateng_data/test_.csv','feateng_data/hist.csv','feateng_data/train.csv','feateng_data/test.csv')
    # generate_full_train_and_test_set('feateng_data/train_set_.pkl','feateng_data/test_set_.pkl','feateng_data/input_data/train_set2.pkl','feateng_data/input_data/test_set2.pkl')
    # with open('feateng_data/input_data/train_set2.pkl','rb') as f:
    #     x_train,y_train, x_theme_hist, x_hist = pkl.load(f)
    # # print(set(x_train[:,-1]))
    # print(len(x_train))
    # with open('feateng_data/input_data/test_set2.pkl','rb') as f:
    #     x_test,y_train, x_theme_hist, x_hist = pkl.load(f)
    # # print(set(x_train[:,-1]))
    # print(len(x_test))
    # print(len(set(np.concatenate([x_train,x_test],axis=0)[:,0])))

    # generate_train_and_test_set('train','feateng_data/train.csv','feateng_data/input_data/train_set.pkl',None)
    # generate_train_and_test_set('test','feateng_data/test.csv','feateng_data/input_data/test_set.pkl','feateng_data/input_data/test_set_small.pkl')
    # split_train_domain('feateng_data/train.csv',['feateng_data/input_data/train_d1.pkl','feateng_data/input_data/train_d2.pkl','feateng_data/input_data/train_d3.pkl'])
    # split_test_domain('feateng_data/input_data/test_set2.pkl',['feateng_data/input_data/test_d1.pkl','feateng_data/input_data/test_d2.pkl','feateng_data/input_data/test_d3.pkl'])
    # split_test_domain('feateng_data/input_data/train_set2.pkl',['feateng_data/input_data/train_d1.pkl','feateng_data/input_data/train_d2.pkl','feateng_data/input_data/train_d3.pkl'])
    # generate_hist_dict


    