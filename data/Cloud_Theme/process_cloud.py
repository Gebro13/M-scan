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

DATA_DIR = ''

def remap_all_data(file_in,file_out):

    print('start remapping')
    data = pd.read_csv(file_in)

    for col in data:
        print('col names:{} size:{}'.format(col,data[col].unique().size))


    remap_num = 1
    remapped_dict = {}
    data = data[['user_id','theme_id','item_id','leaf_cate_id','cate_level1_id','reach_time']]
    data = data.to_numpy()
    for j in tqdm(range(len(data[0])-1)):
        for i in range(len(data)):
            if data[i][j] not in remapped_dict:
                remapped_dict[data[i][j]] = remap_num
                data[i][j] = remap_num
                remap_num+=1
            else:
                data[i][j] = remapped_dict[data[i][j]]

    remap_data = pd.DataFrame(data,columns = ['user_id','theme_id','item_id','leaf_cate_id','cate_level1_id','reach_time'])
    
    remap_data.to_csv(file_out,index=False)
    print(remap_data)
    print('remap finished')
    return remap_data

def generate_dicts(file_in):

    data = pd.read_csv(file_in)
    data = data[['item_id','leaf_cate_id','cate_level1_id']]
    data = data.to_numpy()
    print('generating item dict')
    item_dict = {}
    error_cnt=0
    for i in tqdm(range(len(data))):
        item = data[i][0]
        features = [data[i][1],data[i][2]]
        if item in item_dict:
            if item_dict[item] != features:
                error_cnt+=1
        else:
            item_dict[item] = features
    print(error_cnt)
    with open('feateng_data/item_dict.pkl','wb') as f:
        pkl.dump(item_dict,f)
    
    print('generating theme item dict')
    
    data_all = pd.read_csv(file_in)
    data = data_all[['item_id','theme_id']].drop_duplicates()
    print(len(data))
    theme_item_pairs = data.groupby('theme_id')['item_id'].apply(list).reset_index()
    theme_item_dict = dict(zip(theme_item_pairs['theme_id'], theme_item_pairs['item_id']))
    with open('feateng_data/theme_item_dict.pkl','wb') as f:
        pkl.dump(theme_item_dict,f)
    
    lengths = []
    for theme in theme_item_dict.keys():
        lengths.append(len(theme_item_dict[theme]))
    
    print(len(lengths))
    print(np.mean(lengths))

    print('dicts generation finished')

def generate_negative_samples(file_in,file_out):

    with open('feateng_data/theme_item_dict.pkl','rb') as f:
        theme_item_dict = pkl.load(f)

    data = pd.read_csv(file_in)
    data = data.sort_values(by='reach_time',ascending=True)
    data = data[['user_id','theme_id','item_id']]
    data = data.to_numpy()
    data_all = []
    for row in tqdm(data):
        uid,theme,item = row
        items_in_theme = theme_item_dict[theme]
        data_all.append([uid,theme,item,1])
        neg_items = np.random.choice(items_in_theme,2)
        for neg_item in neg_items:
            while(neg_item == item):
                neg_item = np.random.choice(items_in_theme)
            data_all.append([uid,theme,neg_item,0])

    data_all = pd.DataFrame(data_all,columns = ['user_id','theme_id','item_id','click'])
    data_all.to_csv(file_out,index=False)
    print(data_all)
    

def generate_user_behaviors(file_in):

    data_all = pd.read_csv(file_in)
    data_all = data_all.sort_values(by='reach_time',ascending=True)
    data_all = data_all[['user_id','theme_id','item_id']]
    user_behavs_dict = {}
    user_theme_behavs_dict = {}
    fake_time = 0
    print(len(data_all))
    for row in tqdm(data_all.itertuples(index=False)):
        fake_time+=1
        user, item, theme,click = row[0], row[2], row[1], 1
        
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
    
    with open('feateng_data/user_behavs_dict.pkl','wb') as f:
        pkl.dump(user_behavs_dict,f)
    with open('feateng_data/user_theme_behavs_dict.pkl','wb') as f:
        pkl.dump(user_theme_behavs_dict,f)


    

def dataset_statistics():


    t1 = time.time()
    with open('feateng_data/user_behavs_dict.pkl','rb') as f:
        user_behavs_dict = pkl.load(f)
    with open('feateng_data/user_theme_behavs_dict.pkl','rb') as f:
        user_theme_behavs_dict = pkl.load(f)
        

    lengths =[]
    cnt=0
    user_cnt=0
    for uid in user_behavs_dict.keys():
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
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.xlim(left = 0,right=100)

    plt.show()
    plt.savefig('1.png', dpi=300, bbox_inches='tight')


    lengths = []
    cold_cnt=0
    cnt=0
    multi_theme_user_cnt = 0
    for uid in tqdm(user_theme_behavs_dict.keys()):
        if len(user_theme_behavs_dict[uid].keys()) >1:
            multi_theme_user_cnt +=1
        for theme in user_theme_behavs_dict[uid].keys():
            if len(user_theme_behavs_dict[uid][theme]) <5:
                cold_cnt+=1
            cnt+=1
            lengths.append(len(user_theme_behavs_dict[uid][theme]))
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
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.xlim(left = 0,right=100)

    plt.show()
    
    plt.savefig('4.png', dpi=300, bbox_inches='tight')

    print(time.time()-t1)
    
    print(time.time()-t1)

def train_test_split():

    with open('feateng_data/user_behavs_dict.pkl','rb') as f:
        user_behavs_dict = pkl.load(f)
    with open('feateng_data/user_theme_behavs_dict.pkl','rb') as f:
        user_theme_behavs_dict = pkl.load(f)
    
    
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
    theme_lengths = []
    all_lengths = []
    
    for uid in tqdm(user_theme_behavs_dict.keys()):
        
        cnt+=1
        for theme,theme_behavs in user_theme_behavs_dict[uid].items():
           
            
            if len(theme_behavs)>1 :
        
                theme_behavs =np.array(theme_behavs)
                all_behavs = np.array(user_behavs_dict[uid])
                # print(len(all_behavs))
                avail_behavs = theme_behavs[1:]
                length = len(avail_behavs)
                for i in range(int(0.8*length)):
                    behav = avail_behavs[i]
                    # 同theme的behavior
                    item,click,time = behav
                    theme_behavs_chosen = theme_behavs[theme_behavs[:,2]<time]
                    
                    theme_behavs_chosen = theme_behavs_chosen[theme_behavs_chosen[:,1]==1]
                    
                    # print(len(theme_behavs_chosen))
                    
                    theme_lengths.append(len(theme_behavs_chosen[-5:]))
                    theme_behavs_chosen = padding(theme_behavs_chosen,5)
                    # print(theme_behavs_chosen)
                    # 不同theme的behavior
                    
                    # lengths.append(len(theme_behavs_chosen))
                    behavs_chosen = all_behavs[all_behavs[:,2]<time]
                    all_lengths.append(len(behavs_chosen[-10:]))
                    behavs_chosen = padding(behavs_chosen,10)
            
                    x_train.append([uid,item,theme])
                    y_train.append(click)
            
                    x_theme_hist_train.append(theme_behavs_chosen)
                    x_hist_train.append(behavs_chosen)   
                   
                for i in range(int(0.8*length),int(length)):
                    behav = avail_behavs[i]
                    # 同theme的behavior
                    item,click,time = behav
                    theme_behavs_chosen = theme_behavs[theme_behavs[:,2]<time]
                    
                    theme_behavs_chosen = theme_behavs_chosen[theme_behavs_chosen[:,1]==1]
                    
                    theme_behavs_chosen = padding(theme_behavs_chosen,5)
                     # 不同theme的behavior
                    
                    # lengths.append(len(theme_behavs_chosen))
                    behavs_chosen = all_behavs[all_behavs[:,2]<time]
                    behavs_chosen = padding(behavs_chosen,10)
                    x_test.append([uid,item,theme])
                    y_test.append(click)
                    x_theme_hist_test.append(theme_behavs_chosen)
                    x_hist_test.append(behavs_chosen)   
            
    print(np.mean(theme_lengths))
    print(np.mean(all_lengths))
    print(cnt)
    print(len(x_train))
    print(len(x_test))
    x_train,y_train,x_theme_hist_train,x_hist_train = np.array(x_train),np.array(y_train),np.array(x_theme_hist_train),np.array(x_hist_train)
    x_test,y_test,x_theme_hist_test,x_hist_test = np.array(x_test),np.array(y_test),np.array(x_theme_hist_test),np.array(x_hist_test)

    with open('feateng_data/train_set_.pkl','wb') as f:
        pkl.dump([x_train,y_train,x_theme_hist_train,x_hist_train],f)
    
    with open('feateng_data/test_set_.pkl','wb') as f:
        pkl.dump([x_test,y_test,x_theme_hist_test,x_hist_test],f)

def generate_negative_samples2():

    with open('feateng_data/train_set_.pkl','rb') as f:
        x_train, y_train, x_theme_hist_train, x_hist_train = pkl.load(f)

    with open('feateng_data/test_set_.pkl','rb') as f:
        x_test,y_test,x_theme_hist_test,x_hist_test = pkl.load(f)

    with open('feateng_data/theme_item_dict.pkl','rb') as f:
        theme_item_dict = pkl.load(f)

    x_train_all = []
    y_train_all = []
    x_theme_hist_train_all = []
    x_hist_train_all = []
    for i in tqdm(range(len(x_train))):
        uid,item,theme = x_train[i]
        items_in_theme = theme_item_dict[theme]
        x_train_all.append([uid,item,theme])
        y_train_all.append(1)
        x_theme_hist_train_all.append(x_theme_hist_train[i])
        x_hist_train_all.append(x_hist_train[i])

        neg_items = np.random.choice(items_in_theme,2)
        for neg_item in neg_items:
            while(neg_item == item):
                neg_item = np.random.choice(items_in_theme)
            x_train_all.append([uid,neg_item,theme])
            y_train_all.append(0)
            x_theme_hist_train_all.append(x_theme_hist_train[i])
            x_hist_train_all.append(x_hist_train[i])
    
    x_test_all = []
    y_test_all = []
    x_theme_hist_test_all = []
    x_hist_test_all = []
    for i in tqdm(range(len(x_test))):
        uid,item,theme = x_test[i]
        items_in_theme = theme_item_dict[theme]
        x_test_all.append([uid,item,theme])
        y_test_all.append(1)
        x_theme_hist_test_all.append(x_theme_hist_test[i])
        x_hist_test_all.append(x_hist_test[i])

        neg_items = np.random.choice(items_in_theme,2)
        for neg_item in neg_items:
            while(neg_item == item):
                neg_item = np.random.choice(items_in_theme)
            x_test_all.append([uid,neg_item,theme])
            y_test_all.append(0)
            x_theme_hist_test_all.append(x_theme_hist_test[i])
            x_hist_test_all.append(x_hist_test[i])
    
    x_train_all,y_train_all,x_theme_hist_train_all,x_hist_train_all = np.array(x_train_all),np.array(y_train_all),np.array(x_theme_hist_train_all),np.array(x_hist_train_all)
    x_test_all,y_test_all,x_theme_hist_test_all,x_hist_test_all = np.array(x_test_all),np.array(y_test_all),np.array(x_theme_hist_test_all),np.array(x_hist_test_all)

    with open('feateng_data/train_set_.pkl','wb') as f:
        pkl.dump([x_train_all,y_train_all,x_theme_hist_train_all,x_hist_train_all],f)
    
    with open('feateng_data/test_set_.pkl','wb') as f:
        pkl.dump([x_test_all,y_test_all,x_theme_hist_test_all,x_hist_test_all],f)

    
def generate_full_train_and_test_set(train_set_,test_set_,train_set,test_set):
    print(1)
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
            x_full.append([uid]+[item]+item_dict[item]+[domain])
        for theme_behavs in tqdm(x_theme_hist):
            theme_behavs_full = []
            for theme_behav in theme_behavs:
                item,click,time = theme_behav
                if item in item_dict:
                    theme_behavs_full.append([item]+item_dict[item]+[click])
                else:
                    theme_behavs_full.append([0]*4)
            x_theme_hist_full.append(theme_behavs_full)
        for behavs in tqdm(x_hist):
            behavs_full = []
            for behav in behavs:
                item,click,time = behav
                if item in item_dict:
                    behavs_full.append([item]+item_dict[item]+[click])
                else:
                    behavs_full.append([0]*4)
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
        
def split_train_domain(train_file_in, train_domain_files):
    with open(train_file_in,'rb') as f:
        x_train,y_train,x_theme_hist,x_hist = pkl.load(f)
    # chosen_items = [(775725, 16848), (775649, 21045), (775643, 23124)]
    chosen_items = [(775849, 1224), (775851, 1227), (775903, 1248)]
    for domain,file_out in tqdm(zip(chosen_items,train_domain_files)):
        domain = domain[0]
        print(domain)
        domain_idx = x_train[:,-1] == domain
        print(sum(domain_idx))
        x_train_domain,y_train_domain,x_theme_hist_domain, x_hist_domain = x_train[domain_idx],y_train[domain_idx],x_theme_hist[domain_idx],x_hist[domain_idx]

        with open(file_out,'wb') as f:
            pkl.dump([x_train_domain,y_train_domain,x_theme_hist_domain, x_hist_domain],f,protocol=4)
    
def split_test_domain(test_file_in,test_domain_files):
    with open(test_file_in,'rb') as f:
        x_test,y_test,x_theme_hist,x_hist = pkl.load(f)
    print(x_test)
    domains = set(x_test[:,-1])
    counter = Counter(x_test[:,-1])
    sorted_items = sorted(counter.items(), key=lambda x: x[1])
    chosen_items = sorted_items[-200:-197]
    print(chosen_items)

    for domain,file_out in tqdm(zip(chosen_items,test_domain_files)):
        domain = domain[0]
        print(domain)
        domain_idx = x_test[:,-1] == domain
        print(sum(domain_idx))
        x_test_domain,y_test_domain,x_theme_hist_domain, x_hist_domain = x_test[domain_idx],y_test[domain_idx],x_theme_hist[domain_idx],x_hist[domain_idx]

        with open(file_out,'wb') as f:
            pkl.dump([x_test_domain,y_test_domain,x_theme_hist_domain, x_hist_domain],f,protocol=4)
        

if __name__ == '__main__':
    # remap_all_data('raw_data/theme_click_log.csv','feateng_data/theme_click_log_remapped.csv')
    # data = pd.read_csv('raw_data/theme_click_log.csv')
    # print(data['item_id'].unique().size)


    # generate_dicts('feateng_data/theme_click_log_remapped.csv')
    # generate_negative_samples('feateng_data/theme_click_log_remapped.csv','feateng_data/data.csv')
    # generate_user_behaviors('feateng_data/data.csv')
    # generate_user_behaviors('feateng_data/theme_click_log_remapped.csv')
    # dataset_statistics()
    # print(1)
    # train_test_split()
    # generate_negative_samples2()
    # generate_full_train_and_test_set('feateng_data/train_set_.pkl','feateng_data/test_set_.pkl','feateng_data/input_data/train_set2.pkl','feateng_data/input_data/test_set2.pkl')
    # split_test_domain('feateng_data/input_data/test_set2.pkl',['feateng_data/input_data/test_d4.pkl','feateng_data/input_data/test_d5.pkl','feateng_data/input_data/test_d6.pkl'])
    # split_train_domain('feateng_data/input_data/train_set2.pkl',['feateng_data/input_data/train_d4.pkl','feateng_data/input_data/train_d5.pkl','feateng_data/input_data/train_d6.pkl'])
    # generate_hist_dict

    with open('feateng_data/input_data/train_set2.pkl','rb') as f:
        x_train,y_train, x_theme_hist, x_hist = pkl.load(f)
    # print(set(x_train[:,-1]))
    print(len(x_train))
    with open('feateng_data/input_data/test_set2.pkl','rb') as f:
        x_test,y_train, x_theme_hist, x_hist = pkl.load(f)
    # print(set(x_train[:,-1]))
    print(len(x_test))
    print(len(set(np.concatenate([x_train,x_test],axis=0)[:,0])))

    