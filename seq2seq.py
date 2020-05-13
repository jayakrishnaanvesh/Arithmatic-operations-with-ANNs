from random import seed
from random import randint
import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM,RepeatVector,TimeDistributed
import pandas as pd

def generate_random_examples(n,min,max):
    x_train=[]
    y_train=[]
    for i in range(n):
        x1=randint(min,max)
        x2=randint(min,max)
        y=x1*x2
        x_train.append(str(x1)+"*"+str(x2))
        y_train.append(str(y))
    return x_train,y_train

arr=['0', '1', '2', '3', '4', '5', '6', '7', '8','9','+','*']
X=pd.get_dummies(arr)
no_chars=len(arr)

def get_one_hot_encoding(ar):
    op=[]
    for s in ar:
        r=[]
        for i in range(len(s)):
            y=(X[s[i]]).tolist()
            r.append(y)
        if(len(r)<6):
            r.insert(0,(X['0']).tolist())
        r=np.array(r)
        op.append(r)
    return np.array(op)

def decode_onehot(op):
    result=[]
    for output in op:
        m=[]
        output=pd.DataFrame(output).idxmax(axis=1)
        for i in output:
            m.append(arr[i])
        m=' '.join(map(str, m))
        result.append(m) 
    return result

'''
encoding=get_one_hot_encoding(['1+2','5+6'])
decoding=decode_onehot(encoding)
print(decoding)
'''
def gen_model():
    model = Sequential()
    model.add(LSTM(300, input_shape=(7,no_chars)))
    model.add(RepeatVector(6))
    model.add(LSTM(150, return_sequences=True))
    model.add(TimeDistributed(Dense(no_chars, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    x, y = generate_random_examples(100000, 100, 999)
    #print(len(x))
    x=get_one_hot_encoding(x)
    y=get_one_hot_encoding(y)
    model.fit(x, y, epochs=20, batch_size=32)
    model.save('my_model.h5')

def run_model():
    model=load_model('my_model.h5')
    x, y = generate_random_examples(5, 100, 999)
    #print(len(x))
    print(x)
    x=get_one_hot_encoding(x)
    result = model.predict(x, batch_size=2, verbose=0)
    #print(result.shape)
    print(decode_onehot(result))
    print(y)

#gen_model()
run_model()
