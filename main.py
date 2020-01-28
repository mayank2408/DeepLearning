import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys, getopt
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

def logic(num):
    if (num%15==0):
        return 'FizzBuzz'
    elif (num%5==0):
        return 'Buzz'
    elif (num%3==0):
        return 'Fizz'
    else : 
        return num

def logic2(filename): #uses global neural network
    f = open("Software2.txt", "w")
    q=read_input(filename)
    a=get_input(filename)
    for i,x in enumerate(a):
        temp=net(torch.tensor(x).float())
        out=(np.argmax(temp.detach().numpy(),axis=0))
        if(out==0):
            out= 'Fizz'
        elif (out==1):
            out= 'Buzz'
        elif (out==2):
            out='FizzBuzz'
        else :
            out= q[i]
        f.write(str(out)+'\n')
    f.close()
print("Name: Mayank Gupta")
print("Email Id: mayankg@iisc.ac.in")
print("SR No.: 17112")
print("Course: M.Tech AI")
argv=sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, '', ['help', 'test-data='])
    if not opts:
        print ('wrong input')
        print('python main.py --test-data <test file>')
except getopt.GetoptError as e:
    print (e)
    print('python main.py --test-data <test file>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ('-h', '--help'):
        print('python main.py --test-data <test file>')
        sys.exit(2)
    else:
        net=pickle.load(open("model/Software2","rb"))
        f = open("Software1.txt", "w")
        a=read_input(arg)
        for i in a:f.write(str(logic(i))+'\n')
        f.close()
        logic2(arg)
