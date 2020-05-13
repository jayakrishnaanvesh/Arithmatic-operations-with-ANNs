from random import seed
from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
import numpy as np
def generate_random_examples(n,min,max,train):
    seed(1)
    x_train=[]
    y_train=[]
    for i in range(n):
        x1=randint(min,max)
        x2=randint(min,max)
        y=x1+x2
        if(train==False):
            print(x1,x2,y)
        x1=get_binary(x1)
        x2=get_binary(x2)
        x_train.append(np.append(x1,x2))
        y_train.append(y)
    x_train= np.array(x_train)
    y_train =np.array(y_train)
    #print(x_train,y_train)
    return x_train,y_train

def get_binary(number):
  l=18
  res = [int(i) for i in list('{0:0b}'.format(number))] 
  l_res=len(res)
  for i in range(0,l-l_res):
    res.insert(0,0)
  return np.array(res)


def train_model(num_layers,num_neurons):
  model = Sequential()
  model.add(Dense(36,use_bias=True, input_dim=36))
  for i in range(num_layers):
    model.add(Dense(num_neurons,use_bias=True))
  model.add(Dense(1,use_bias=True))
  model.compile(loss='mean_squared_error', optimizer='adam')
  x , y =generate_random_examples(n_example,min,max,True)
  #print(x.shape,y.shape)
  hist=model.fit(x,y,batch_size=16,epochs=15,shuffle=True,verbose=2)
  x ,y = generate_random_examples(5,min,max,False)
  result = model.predict(x, batch_size=2, verbose=0)
  return result,hist.history['loss'][-1]





n_example=10000
min=1
layers=10
max=10000

'''
for i in range(layers):
  print("################number of layers=",i)
  result,final_loss=train_model(i,36)
  print("layer:",i,"loss:", final_loss )
  if(final_loss<0.1):
    break
'''
'''
for i in range(10):
  result,final_loss=train_model(4,18*i)
  print("num_neurons:",18*i,"loss:", final_loss )
  if(final_loss<0.01):
    break
'''

result, final_loss=train_model(2,36)
print(result)