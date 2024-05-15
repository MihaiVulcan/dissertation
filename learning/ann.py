import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
import torch.optim as optim

from utils.data import Data 
from visualization import plot_actual_predicted

class Network(nn.Module):
    def __init__(self, input_dim, hidden_layer, output_dim):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, output_dim)


    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

def train(type, epochs, hidden_layer_size, leaning_rate, features, output_features, data_file, mutation_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    #read data
    data = pd.read_csv(data_file)
    X = data.loc[:, features]
    y = data.loc[:, output_features]

    
    #normalize
    norm_X = pd.DataFrame
    og_dataset_min = X.min()
    og_dataset_max = X.max()
    if type == "triangle":
        norm_X = X
    elif type == "credit":
        norm_X = (X-X.min())/(X.max()-X.min())
        norm_X['preffed_custommer'] = X["preffed_custommer"]

    y = data.loc[:, output_features]
    X_train,X_test,y_train,y_test = train_test_split(norm_X, y, test_size = 0.2)

    traindata = Data(X_train , y_train)
    testdata = Data(X_test, y_test)

    batch_size = 4
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)

    model = Network(len(features),hidden_layer_size,1).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=leaning_rate)

    accuracies = []
    mares = []
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
                print(f'[{epoch}, {i:5d}] loss: {running_loss/((i+1)):.5f}')
        
        if(epoch%10==0):
            error, errorabs, correct, total = 0, 0, 0, 0
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
                    correct += torch.sum(torch.eq(labels, torch.round(outputs)))
            print(f'EpochNo:{epoch} MRE: {error/total}\n MARE: {errorabs/total}\n accuracy:{correct/total}')
            accuracies.append(correct/total)
            mares.append(errorabs/total)
        print(" ")


   
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=2)
    error, errorabs, correct, total = 0, 0, 0, 0
    all_outputs, all_labels= [], []
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
            correct += torch.sum(torch.eq(labels, torch.round(outputs)))
            all_labels.append(labels.cpu())
            all_outputs.append(outputs.cpu())
            print(f'outputs: {outputs}')
            print(f'labels: {labels}')
            print(f'error: {torch.sum(torch.sub(outputs, labels))/4}')
            print(f'errorabs: {torch.sum(torch.abs(torch.sub(outputs, labels)))/4}')
            
    print(f'MRE: {error/total}\n MARE: {errorabs/total}\n accuracy:{correct/total}')
    #print("accuracies: ")
    #print(accuracies)
    #print("MAREs: ")
    #print(mares)

    #mutation testing
    if type == "credit":
        data = pd.read_csv(mutation_file)
        X = data.loc[:, features]
        y = data.loc[:, ["cat", "actual_cat"]]
        norm_X = (X-og_dataset_min)/(og_dataset_max-og_dataset_min)
        norm_X['preffed_custommer'] = X["preffed_custommer"]
        validation_data = Data(norm_X, y)
        validationLoader = DataLoader(validation_data, batch_size=1, shuffle=False, num_workers=2)
        correct_as_correct, correct_as_incorrect, incorrect_as_correct, incorrect_as_incorrect = 0, 0, 0, 0
        with torch.no_grad():
            for data in validationLoader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # calculate output by running through the network
                outputs = model(inputs)
                # check results
                outputs = outputs.round()
                #Correct records classified as Correct
                if outputs[0][0] == labels[0][1] and labels[0][0] == labels[0][1]:
                    correct_as_correct += 1
                #Correct records classified as Incorrect (Faulty)
                if outputs[0][0] != labels[0][1] and labels[0][0] == labels[0][1]:
                    correct_as_incorrect += 1
                #Incorrect (Faulty) records classified as Correct
                if outputs[0][0] != labels[0][1] and outputs[0][0] == labels[0][0]:
                    incorrect_as_correct += 1
                #Incorrect (Faulty) records classified as Incorrect (Faulty)
                if outputs[0][0] != labels[0][0] and labels[0][1] != labels[0][0]:
                    incorrect_as_incorrect += 1
        print("correct as correct: ")
        print(correct_as_correct)
        print("correct as incorect: ")
        print(correct_as_incorrect)
        print("incorect as incorect: ")
        print(incorrect_as_correct)
        print("incorect as incorect: ")
        print(incorrect_as_incorrect)
        print("sum")
        print(correct_as_correct+correct_as_incorrect+incorrect_as_correct+incorrect_as_incorrect)
        print("model_is_right")
        print(correct_as_correct+incorrect_as_incorrect)
        print("model_is_NOT_right")
        print(incorrect_as_correct+correct_as_incorrect)

        #visualize
        plot_actual_predicted(all_labels, all_outputs)



def main(type, epochs, hidden_layer_size, leaning_rate, features, output_features, data_file, mutation_file):
    if __name__ == '__main__':
        freeze_support()
        train(type, epochs, hidden_layer_size, leaning_rate, features, output_features, data_file, mutation_file)

#triangle
main("triangle", 100, 24, 0.001, ['side_a','side_b','side_c'], ['triangle_type'], './dataset_generator/triangle_dataset/data/dataset.csv', '')

#bank_credit
#main("credit", 100, 50, 0.001, ['age','yearly_salary','year_wanted','preffed_custommer'], ['cat'], './dataset_generator/credit_dataset/data/dataset.csv', './dataset_generator/credit_dataset/fault_injection/data/dataset.csv')