import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
import torch.optim as optim

from utils.data import Data 

class Network(nn.Module):
    def __init__(self, input_dim, hidden_layer, output_dim):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    #read data
    data = pd.read_csv('./dataset_generator/triangle_dataset/data/dataset_multioutput.csv')
    features = ['side_a','side_b','side_c']
    labels = ['triangle_type1', 'triangle_type2', 'triangle_type3', 'triangle_type4']
    X = data.loc[:, features]
    y = data.loc[:, labels]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2)

    traindata = Data(X_train , y_train)
    testdata = Data(X_test, y_test)

    batch_size = 4
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)

    model = Network(3,24,4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 50
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward propagation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward propagation 
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
            # display statistics
            if(i%20==0):
                print(f'[{epoch}, {i:5d}] loss: {running_loss/2000:.5f}')
        error, errorabs, total = 0, 0, 0
        if(epoch%10==0):
            testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=2)
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    # calculate output by running through the network
                    outputs = model(inputs)
                    # update results
                    total += labels.size(0)
                    error+= torch.sum(torch.sub(outputs, labels))
                    errorabs+= torch.sum(torch.abs(torch.sub(outputs, labels)))
            print(f'EpochNo:{epoch} MRE: {error/total}\n MARE: {errorabs/total}')
        print(" ")


   
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=2)
    error, errorabs, total = 0, 0, 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # calculate output by running through the network
            outputs = model(inputs)
            # update results
            total += labels.size(0)
            error+= torch.sum(torch.sub(labels, outputs))
            errorabs+= torch.sum(torch.abs(torch.sub(labels, outputs)))
            print(f'outputs: {outputs}')
            print(f'labels: {labels}')
            print(f'error: {torch.sum(torch.sub(outputs, labels))/4}')
            print(f'errorabs: {torch.sum(torch.abs(torch.sub(outputs, labels)))/4}')
    print(f'MRE: {error/total}\n MARE: {errorabs/total}')

def main():
    if __name__ == '__main__':
        freeze_support()
        train()

    # criterion = nn.CrossEntropyLoss()
    # out = torch.tensor([[0.0075], [0.0053], [0.2250], [0.2258]], dtype=torch.float)
    # target = torch.tensor([[3.], [2], [1.], [1.]], dtype=torch.float)
    # print(out)
    # print(torch.max(out, dim=1).values)
    # print(criterion(torch.max(out, dim=1).values, torch.max(target, dim=1).values))
    # print(criterion(torch.tensor([0.0075, 0.0053]), torch.tensor([3., 2.])))

main()