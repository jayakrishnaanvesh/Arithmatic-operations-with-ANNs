from random import seed
from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
import numpy as np
def generate_random_examples(n,min,max):
    seed(1)
    x_train=[]
    y_train=[]
    for i in range(n):
        x1=randint(min,max)
        x2=randint(min,max)
        y=x1+x2
        x_train.append([x1,x2])
        y_train.append(y)
    x_train= np.array(x_train)/float(max*2)
    y_train =np.array(y_train)/float(max*2)
    return x_train,y_train



def train_model(num_layers,num_neurons):
    model = Sequential()
    model.add(Dense(num_neurons,use_bias=True, input_dim=2))
    for i in range(num_layers):
        model.add(Dense(num_neurons,use_bias=True))
    model.add(Dense(1,use_bias=True))
    model.compile(loss='mean_squared_error', optimizer='adam')
    x , y =generate_random_examples(n_example,min,max)
    hist=model.fit(x,y,batch_size=2,epochs=5,shuffle=True,verbose=0)
    x  = np.array([[12,23],[197,283],[15,265],[1999,4444],[555,23]])/(max*2)
    #print(x)
    result = model.predict(x, batch_size=2, verbose=0)
    predicted = [r*max*2 for r in result]
    return predicted,hist.history['loss'][-1]


n_example=10000
min=1
max=10000
layers=10

'''
for i in range(layers):
  print("################number of layers=",i)
  result,final_loss=train_model(i,4)
  print("layer:",i+10,"loss:", final_loss )
  if(final_loss<0.01):
    break
'''
'''
for i in range(1,10):
  result,final_loss=train_model(12,4*i)
  print("num_neurons:",4*i,"loss:", final_loss )
  if(final_loss<0.01):
    break
'''

result,final_loss=train_model(0,4)

print(result)