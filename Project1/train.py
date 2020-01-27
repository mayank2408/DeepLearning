import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
def read_input(filename):
    f=open(filename,"r")
    p = np.array(f.read().split(),dtype=int)
    return p

def decimal_to_binary(n):  
    return bin(n).replace("0b", "") 

def convert(num):
    d=decimal_to_binary((num))
    d='0'*(10-len(d))+d
    return d

def get_input(filename):
    a=read_input(filename)
    b=[]
    for x in a:
        b.append(list(convert(int(x))))
    b=np.array(b,dtype=int)
    return b

def convert_output(x):
    if(x=='Fizz' or x=='fizz'):
        return 0
    elif (x=='Buzz'or x=='buzz'):
        return 1
    elif (x=='FizzBuzz' or x=='fizzbuzz'):
        return 2
    else :
        return 3

def read_output(filename):
    f=open(filename,"r")
    p = np.array(f.read().split())
    return p

def get_output(filename):
    d=[]
    a=read_output(filename)
    for x in a:
        d.append(convert_output(x))
    d=np.array(d)
    return d

def accuracy(test_input,test_output):
    test_data=read_input(test_input)
    test_in=[]
    for x in test_data:
        test_in.append(list(convert(int(x))))
    test_in=np.array(test_in,dtype=int)
    with torch.no_grad():
        inpuuut=torch.tensor(test_in)
        out=net(inpuuut.float())
        yo=(np.argmax(out.detach().numpy(),axis=1))
    test_o=[]
    tem=read_output(test_output)
    for x in tem:
        test_o.append(convert_output(x))
    test_o=np.array(test_o)
    acc=(yo-test_o)==0
    print('accuracy=',sum(acc)/len(acc))
    print('accuracy fizz:',1-sum((test_o==0)!=(yo==0))/len(test_o==0))
    print('accuracy buzz:',1-sum((test_o==1)!=(yo==1))/len(test_o==1))
    print('accuracy fizzbuzz:',1-sum((test_o==2)!=(yo==2))/len(test_o==2))
    return sum(acc)/len(acc)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1=nn.Linear(10,15)
        self.fc2=nn.Linear(15,8)
        self.fc3=nn.Linear(8,4)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net=NeuralNet()
w = torch.tensor([0.6,1,2,0.2],dtype=torch.float32)
criterion=nn.CrossEntropyLoss(weight=w)

lr=0.01
optimizer=optim.Adam(net.parameters(),lr,weight_decay=0.001)

epochs=3000
batch_size=64
num=int(900/batch_size)

b=get_input('train_input.txt')
d=get_output('train_output.txt')

order=np.arange(900)
for i in range(epochs):
    '''if (epochs%3000)==2999:
        lr=lr/3
    optimizer=optim.Adam(net.parameters(),lr)'''
    running_loss=0.0
    np.random.shuffle(order)
    train_input_copy=b[order]
    train_output_copy=d[order]
    
    for j in range(num-1):
        input_batch=torch.tensor(train_input_copy[j*batch_size:(j+1)*batch_size,:])
        output_batch=torch.tensor(train_output_copy[j*batch_size:(j+1)*batch_size])
        #print(input_batch)
        #print(output_batch)
        optimizer.zero_grad()
        output=net(input_batch.float())
        
     
        loss=criterion(output,output_batch)
        loss.backward()
        optimizer.step()
        '''if j % 400 == 99:
            print(np.argmax(output.detach().numpy(),axis=1))
            print(output_batch)'''
        
        running_loss +=loss.item()
    if i % 10 == 9:
        print('[%d, %5d] loss: %.5f' %
              (epochs , i + 1, running_loss / 900))
        running_loss = 0.0
print(accuracy('test_input.txt','test_output.txt'))
#pickle.dump(net, open("model/software2", 'wb'))