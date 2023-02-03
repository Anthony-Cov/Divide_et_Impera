# Работа с данными
import pandas as pd
import numpy as np
# Процесс выполнения
from tqdm.notebook import tqdm,trange
# PyTorch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import torch.nn as nn
import torch


class LSTM_cat_model(nn.Module):

    def __init__(self, input_size=32, hidden_size=128, to_pred=7,
                 dropout_inside=0.1,dropout_outside=0.1):
        '''
        Параметры:
          input_size -- количество признаков во входных данных
          hidden_size -- размер скрытого слоя для LSTM
          to_pred -- какое кол-во timesteps предсказывать
          dropout_inside -- значение дропаута внутри LSTM
          dropout_outside -- значение дропаута после полносвязного слоя
        '''
        super(LSTM_cat_model, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            dropout=dropout_inside,
                            batch_first=True,
                            bidirectional=False
                          )
        self.linear = nn.Linear(in_features=self.hidden_size, 
                                out_features=16)
        self.act = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_outside)
        self.linear1 = nn.Linear(in_features=16, 
                                 out_features=to_pred)

    def forward(self, input):
        '''
        Параметры:
          input -- входные данные, где batch_size на 0-й позиции!
        '''
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero
        out, (h, c) = self.lstm(input)
        lstm_output = h[-1,:,:].view(-1, self.hidden_size)
        
        linear_out = self.linear(lstm_output)
        linear_out = self.act(linear_out)
        linear_out = self.dropout2(linear_out)
        
        linear_out = self.linear1(linear_out)
        
        return linear_out
    
    
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Подправить learning rate.
    Параметры:
      optimizer -- оптимизатор
      shrink_factor -- learning rate умножается на n.
    """
    #print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    #print(f"The new learning rate is {(optimizer.param_groups[0]['lr'],)}")
    

def main_s(train_loader, test_loader, the_model, loss_function, optimizer, epoch_n):
    '''
    Параметры:
      train_loader -- даталоадер для обучающей выборки
      test_loader -- даталоадер для валидационной выборки
      the_model -- сама модель
      loss_function -- функция ошибки
      optimizer -- оптимизатор
      epoch_n -- кол-во эпох
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, test_losses = [], []
    best_loss = 999
    epochs_since_improvement = 0
    checkpoint= {}
    
    # Для каждой эпохи
    for epoch in trange(epoch_n, desc='epoch'):
        batch_losses = []
        
        the_model.train() # "Включить" режим обучения (dropout слой будет работать)
        
        if epochs_since_improvement == 100:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
            adjust_learning_rate(optimizer, 0.8)
                
        # Обучение для каждого батча
        for i, (input, y) in enumerate(train_loader):
            # переносим тензоры на GPU, если можно
            input = input.to(device)
            y = y.to(device)
            
            # пропускаем через модель и получаем предсказания
            preds = the_model(input)
            
            optimizer.zero_grad() # обнуляем градиенты, чтобы не накапливались с предыдущих
            
            loss = loss_function(preds, y) # считаем ошибку
            loss.backward()
            
            optimizer.step() # обновляем веса

            batch_losses.append(loss.item())

        batch_losses = np.array(batch_losses)
        train_losses.append(np.mean(batch_losses))
        #print(f'TRAIN: {epoch} epoch loss: {train_losses[-1]:.4f}', end="")
        
        # Проводим валидацию после эпохи обучения
        # Валидация для каждого батча
        the_model.eval() # "Включить" режим валидации 

        with torch.no_grad(): # вручную отключаем вычисление градиентов

            batch_losses = []

            for i, (input, y) in enumerate(test_loader):
                input = input.to(device)
                y = y.to(device)
                preds = the_model(input)
                
                loss = loss_function(preds, y) 
                batch_losses.append(loss.item())

        batch_losses = np.array(batch_losses)

        
        test_losses.append(np.mean(batch_losses))
        recent_loss = test_losses[-1]
        #print(f'___TEST: {epoch} epoch loss: {recent_loss:.4f}', end="")
        
        # Корректируем epochs_since_improvement в зависимости от метрики
        best_loss = min(recent_loss, best_loss)
        if recent_loss > best_loss:
            epochs_since_improvement += 1
            #print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
        else:
            epochs_since_improvement = 0
            checkpoint = {'model': the_model.state_dict(),
                          'optimizer' : optimizer.state_dict()}
            #print("Saving")
            
    #print(f'Train: {train_losses[-1]}, Test: {test_losses[-1]}')
    return train_losses, test_losses, checkpoint


class TransactionsDataset(Dataset):
      def __init__(self,x,y):
        self.x = torch.tensor(x, dtype=torch.float32)#float32
        self.y = torch.tensor(y, dtype=torch.float32)

      def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

      def __len__(self):
        return self.x.shape[0]


def pred(input, the_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    the_model.eval()
    preds = the_model(input)
    
    # переводим предсказания из логитов в нормальный вид
    # можно с сигмоидой и порогом 0.5 или с логитами и порогом 0.
    return (preds > 0).long()
    

def train_model(all_x, all_y, all_test_x, all_test_y, 
                e_n=100, lr=0.0005, cat=0, n_cats=3, n_train=28):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    x_for_train = np.concatenate((all_x[:,:,n_cats:],all_x[:,:,cat].reshape(-1,n_train,1)),axis=2)
    x_for_test = np.concatenate((all_test_x[:,:,n_cats:],all_test_x[:,:,cat].reshape(-1,n_train,1)),axis=2)
    
    # считаем веса
    weights = []
    for class_l in range(2):
        weights.append(x_for_train[:,:,-1][x_for_train[:,:,-1]==class_l].shape[0])
    weights = torch.tensor(weights, dtype=torch.float32)  
    weights = weights[0]/weights[1]
    weights
    
    train_loader = torch.utils.data.DataLoader(TransactionsDataset(x_for_train[:], 
                                                                   all_y[:,:,cat].astype('float32')),
                                               batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)  
    test_loader = torch.utils.data.DataLoader(TransactionsDataset(x_for_test, 
                                                                  all_test_y[:,:,cat].astype('float32')),
                                               batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    the_model = LSTM_cat_model(input_size=5, hidden_size=64, to_pred=7,
                            dropout_inside=0.2,dropout_outside=0.1).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, the_model.parameters()),lr=lr)

    # обучаем!
    train_losses, test_losses, checkpoint = main_s(train_loader, test_loader, the_model, 
                                                                 loss_function, optimizer, epoch_n=e_n)
    
    return the_model, optimizer, train_losses, test_losses, checkpoint, weights


def base_model_data(q, model_after,days,cat=0, n_train=28, n_pred=7, n_features=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #drop date column
    all_test_x = q.iloc[:,:-n_pred*(n_features+1)].drop(q.iloc[:,7:-n_pred*(n_features+1):8], axis=1, inplace=False
                                                       ).values.reshape(-1,n_train,n_features)
    all_test_y = q.iloc[:,-n_pred*(n_features+1):].values.reshape(-1,n_pred,(n_features+1))

    x_for_test = np.concatenate((all_test_x[:,:,3:],all_test_x[:,:,cat].reshape(-1,n_train,1)),axis=2)

    test_loader_n = torch.utils.data.DataLoader(TransactionsDataset(x_for_test, 
                                                                    all_test_y[:,:,cat].astype('float32')),
                                               batch_size=x_for_test.shape[0], shuffle=False, pin_memory=True)
    tests, y_t = next(iter(test_loader_n))

    model_after.eval()
    preds = pred(tests, model_after)
    y = y_t.to(device)

    r=[]
    y_days = all_test_y[:,:,-1].astype('datetime64[D]')
    for week_n, week in enumerate(days):
        pred_wday = preds[week_n]
        y_wday = y[week_n]

        #f1_scores = f1_score(y_wday.detach().cpu(), pred_wday.detach().cpu(), 
                             #zero_division=0)
        r.append([y_wday.detach().cpu().numpy(),
                  pred_wday.detach().cpu().numpy()])
    
    return r


def incr_model_data(q, model_after,days,optimizer_after,weights,with_weights=True,cat=0,
                   n_train=28, n_pred=7, n_features=7, n_cats=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    i_r = []
    for week in days:
        qq = q[(week[0]==q['date+1'])&(q['date+7']==week[-1])]

        all_test_x = qq.iloc[:,:-n_pred*(n_features+1)].drop(qq.iloc[:,n_features:-n_pred*(n_features+1):n_features+1], 
                                                             axis=1, inplace=False).values.reshape(-1,n_train,n_features)
        all_test_y = qq.iloc[:,-n_pred*(n_features+1):].values.reshape(-1,n_pred,(n_features+1))

        x_for_test = np.concatenate((all_test_x[:,:,n_cats:],all_test_x[:,:,cat].reshape(-1,n_train,1)),axis=2)

        loader = torch.utils.data.DataLoader(TransactionsDataset(x_for_test, 
                                                                 all_test_y[:,:,cat].astype('float32')),
                                                   batch_size=7, shuffle=False, pin_memory=True)
        
        # считаем веса
        if with_weights:
            weights = []
            for class_l in range(2):
                weights.append(x_for_test[:,:,-1][x_for_test[:,:,-1]==class_l].shape[0])
            weights = torch.tensor(weights, dtype=torch.float32)  
            weights = weights[0]/weights[1]
            weights
            loss_function = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)
        else: loss_function = nn.BCEWithLogitsLoss().to(device)
            
        x_vals,y,preds = pred_and_train(loader,model_after,loss_function,optimizer_after)
        
        i_r.append([y.reshape(-1).detach().cpu().numpy(),
                    preds.detach().cpu().numpy()])
        
    return i_r


def pred_and_train(loader, the_model, loss_function, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i, (x, y) in enumerate(loader):
        the_model.eval() # "Включить" режим валидации
        
        x = x.to(device)
        y = y.to(device)
        preds_before = pred(x, the_model)
        
        
        the_model.train() # "Включить" режим обучения (dropout слой будет работать)
        
        preds = the_model(x)
        optimizer.zero_grad() # обнуляем градиенты, чтобы не накапливались с предыдущих
        loss = loss_function(preds, y) # считаем ошибку
        loss.backward()
        optimizer.step() # обновляем веса

        #print(f'\nTRAIN loss: {loss.item():.4f}', end="")
        
    return x,y,preds_before