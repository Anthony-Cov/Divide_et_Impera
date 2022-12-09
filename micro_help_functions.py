# Работа с данными
import pandas as pd
import numpy as np


def get_df(df, client_id, k=1, n_cats=3, n_train=28, n_pred=7):
    
    a = pd.DataFrame(df.loc[client_id], 
                 columns=[*[f'event_{i}' for i in range(n_cats)]
                          ,'d_sin','d_cos','m_sin','m_cos'])

    cols = ['d_sin','d_cos','m_sin','m_cos']
    a[cols] = a[cols].apply(lambda x: x.astype('float16'))
    cols = [f'event_{i}' for i in range(n_cats)]
    a[cols] = a[cols].apply(lambda x: x.astype('int8'))

    # каждую k-ую запись
    orig_cols = a.columns

    to_add = []
    for i in range(1,n_train):
          for col in orig_cols[:]:
                to_add.append(a[col].shift(-i).iloc[::k].rename(f'{col}-{i}'))

    for i in range(n_train, n_train+n_pred):
          for col in orig_cols[:n_cats]: #предсказываем факт совершения транзакции
                to_add.append(a[col].shift(-i).iloc[::k].rename(f'{col}+{i-n_train+1}'))
                
    a = pd.concat([a.iloc[::k]]+to_add, axis=1) 
    a.dropna(inplace=True)
    
    return a


def get_splits_by_client(client_id, train_df, test_df, n_cats=3, n_train=28, n_pred=7, n_features=7):

    train_client = get_df(train_df, client_id, k=1, n_train=n_train, n_pred=n_pred)
    test_client = get_df(test_df, client_id, k=1, n_train=n_train, n_pred=n_pred)
    
    all_x = train_client.iloc[:,:-n_pred*n_cats].values.reshape(-1,n_train,n_features)
    all_y = train_client.iloc[:,-n_pred*n_cats:].values.reshape(-1,n_pred,n_cats)

    all_test_x = test_client.iloc[:,:-n_pred*n_cats].values.reshape(-1,n_train,n_features)
    all_test_y = test_client.iloc[:,-n_pred*n_cats:].values.reshape(-1,n_pred,n_cats)
    
    return all_x,all_y,all_test_x,all_test_y


def get_df_full_y_new(df, min_date='2020-07-01',max_date='2021-06-29', n_cats=3,
                     n_train=28,n_pred=7):
    
    a = pd.DataFrame(df, columns=[*[f'event_{i}' for i in range(n_cats)]
                          ,'d_sin','d_cos','m_sin','m_cos'])

    cols = ['d_sin','d_cos','m_sin','m_cos']
    a[cols] = a[cols].apply(lambda x: x.astype('float16'))
    cols = [f'event_{i}' for i in range(n_cats)]
    a[cols] = a[cols].apply(lambda x: x.astype('int32'))
    a['date'] = pd.date_range(min_date,max_date)

    orig_cols = a.columns

    to_add = []
    for i in range(1,n_train):
          for col in orig_cols[:]:
                to_add.append(a[col].shift(-i).rename(f'{col}-{i}'))

    for i in range(n_train, n_train+n_pred):
          for col in orig_cols[:]: #предсказываем 
                to_add.append(a[col].shift(-i).rename(f'{col}+{i-n_train+1}'))
                
    a = pd.concat([a]+to_add, axis=1) 
    a.dropna(inplace=True)
    
    return a


def F1metr(x_real,x_pred): #классы: 1 - positive, O - negative
    '''
    Подсчет F-меры вручную, чтобы F1-score([0,0,0],[0,0,0]) был 1.
    '''
    
    x_pred, x_real= x_pred.astype(int), x_real.astype(int) 
    tp=len(np.where(x_pred[np.where(x_real==1)]==1)[0])
    tn=len(np.where(x_pred[np.where(x_real==0)]==0)[0])
    fp=len(np.where(x_pred[np.where(x_real==0)]==1)[0])
    fn=len(np.where(x_pred[np.where(x_real==1)]==0)[0])
    
    if (tp+fp)*(tp+fn)*tp:
        precision, recall = tp/(tp+fp), tp/(tp+fn)
        f1=2*precision*recall/(precision+recall) 
    else:
        f1=0.
        
    if (tp+tn+fp+fn):
        accuracy=(tp+tn)/(tp+tn+fp+fn)*100
    else:
        accuracy=0.
        
    if accuracy>99.: f1=1
    
    return f1


def apply_metric(metric, array):
        return np.array([[[metric(ys[0],ys[1]) for ys in weeks] for weeks in cats] for cats in array])
    
    
def get_base_inc_arrays(df_2models, metric, col_base='base', col_inc='inc',
                       n_cats=3, n_test_weeks=25, n_pred=7):
    '''
    '''
    # Отдельно выделим массивы для каждой модели
    base_array = np.array([np.vstack(x) for x in df_2models[col_base].to_numpy().reshape(-1)],dtype=object
                         ).reshape(-1,n_cats,n_test_weeks,2,n_pred) # 2 -- number of models

    inc_array = np.array([np.vstack(x) for x in df_2models[col_inc].to_numpy().reshape(-1)],dtype=object
                        ).reshape(-1,n_cats,n_test_weeks,2)
    inc_array = np.array([i.flatten() for i in inc_array.reshape(-1)],dtype=object
                        ).reshape(-1,n_cats,n_test_weeks,2,n_pred)
    
    # Считаем заданную метрику по предсказаниям и реальным данным моделей
    inc_with_metric = apply_metric(metric, inc_array)
    base_with_metric = apply_metric(metric, base_array)
    
    # Считаем "среднюю" метрику для каждой недели, чтобы из трех категорий было одно число
    # "средняя": корень из суммы квадратов метрик по категориям делим на корень трех.
    inc_with_metric_3in1 = np.array([np.sqrt(np.sum(user**2,0))/np.sqrt(3) for user in inc_with_metric])
    base_with_metric_3in1 = np.array([np.sqrt(np.sum(user**2,0))/np.sqrt(3) for user in base_with_metric])
    
    return inc_with_metric, inc_with_metric_3in1, base_with_metric, base_with_metric_3in1