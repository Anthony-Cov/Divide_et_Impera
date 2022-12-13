# Работа с данными
import numpy as np
from scipy.linalg import hankel
# Процесс выполнения
from tqdm.notebook import tqdm,trange
# Tensorflow
import tensorflow
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense, concatenate, Dropout, LSTM


def MakeSet(ser, dim, mem):
    #dim=max(CEmbDim(ser)*2, fwd)
    H=hankel(ser)
    X0=H[:-dim, :dim]
    X=[]
    for i in range(X0.shape[0]-mem-1):
        X.append(X0[i:i+mem, :])  
    X=np.array(X)
    y=H[mem+1:-dim, dim:dim+1]
    return X, y

def fit_lstm(X, y, nb_epoch, n_batch, n_neurons, additional=[],learning_rate=0.00005,verb=1):
    n_batch= n_batch
    tensorflow.keras.utils.set_random_seed(42)
    
    in1 = Input(batch_shape = (n_batch,X.shape[1], X.shape[2]))
    out = LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), 
                stateful=True, return_sequences=False, activation='relu')(in1)
    out = Dense(32, activation='relu')(out)
    out = Dropout(0.1)(out)
    x = Dense(y.shape[1], activation='linear')(out)
    
    model = Model(inputs=[in1], outputs=x)
    fitdat=[X]
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error',metrics=['mape'], optimizer=optimizer,  run_eagerly=True)

    #for i in trange(nb_epoch):
    hist = model.fit(fitdat, y, validation_split=0.2,
                     epochs=nb_epoch, batch_size=n_batch, verbose=verb, shuffle=False)
        #history_loss.append(hist.history['loss'])
        #history_val_loss.append(hist.history['val_loss'])
    #print(model.summary())
    
    return model#,history_loss,history_val_loss

def make_forecast(model, dat, pred, dim, mem, split, imp_num, fwd):
    zfwd=np.array([])
    trg=dat[-(dim+mem+split+1-imp_num):-split+1+imp_num]
    
    for i in range(fwd):
        
        X, y = MakeSet(trg, dim, mem)
        inp=[X[:1]]
        
        z=model.predict(inp, verbose=0)[0]
        zfwd=np.concatenate((zfwd, z))
        trg=np.concatenate((trg[1:], z))
    return zfwd