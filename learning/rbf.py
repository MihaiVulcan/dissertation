import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
import torch.optim as optim

from utils.data import Data 
from torch_rbf import RBF, gaussian
from visualization import plot_actual_predicted, plot_roc_curve

class Network(nn.Module):
    
    def __init__(self, layer_widths, layer_centres, basis_func):
        super(Network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(RBF(layer_widths[i], layer_centres[i], basis_func))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i+1]))
    
    def forward(self, x):
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out

def train_rbf(type, epochs, hidden_layer_size, leaning_rate, return_result):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    features, output_features, data_file, mutation_file = [], [], '',  ''
    if type == "triangle":
        features = ['side_a','side_b','side_c']
        output_features = ['triangle_type']
        data_file = './dataset_generator/triangle_dataset/data/dataset.csv'
        mutation_file = ''
    elif type == "credit":
        features = ['age','yearly_salary','year_wanted','preffed_custommer']
        output_features = ['cat']
        data_file = './dataset_generator/credit_dataset/data/dataset.csv'
        mutation_file = './dataset_generator/credit_dataset/fault_injection/data/dataset.csv'
    elif type == "heart_risk":
        features =  ['gender', 'age', 'bmi', 'exercices', 'stress', 'smoking']
        output_features = ['cat']
        data_file =  './dataset_generator/heart_risk_dataset/data/dataset.csv'
        mutation_file = './dataset_generator/heart_risk_dataset/fault_injection/data/dataset.csv'

    #read data
    data = pd.read_csv(data_file)
    X = data.loc[:, features]
    y = data.loc[:, output_features]

    #normalize
    og_dataset_min = X.min()
    og_dataset_max = X.max()
    norm_X = pd.DataFrame
    if type == "triangle":
        norm_X = X/200
    elif type == "credit":
        norm_X = (X-X.min())/(X.max()-X.min())
        norm_X['preffed_custommer'] = X["preffed_custommer"]
    elif type == "heart_risk":
        norm_X = (X-X.min())/(X.max()-X.min())
        norm_X['gender'] = X["gender"]
        norm_X['smoking'] = X["smoking"]

    #split
    X_train,X_test,y_train,y_test = train_test_split(norm_X, y, test_size = 0.2)

    traindata = Data(X_train , y_train)
    testdata = Data(X_test, y_test)

    batch_size = 4
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)

    layer_widths = [len(features), 1]
    layer_centres = [hidden_layer_size]
    basis_func = gaussian

    model = Network(layer_widths, layer_centres, basis_func).to(device)

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

        error, errorabs, correct, total = 0, 0, 0, 0
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
                    error+= torch.sum(torch.sub(outputs, labels)/labels)
                    errorabs+= torch.sum(torch.abs(torch.sub(outputs, labels)/labels))
                    correct += torch.sum(torch.eq(labels, torch.round(outputs)))
            print(f'EpochNo:{epoch} MRE: {error/total:.5f}\n MARE: {errorabs/total:.5f}\n accuracy:{correct/total:.5f}')
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
            error+= torch.sum(torch.sub(labels, outputs)/labels)
            errorabs+= torch.sum(torch.abs(torch.sub(labels, outputs)/labels))
            correct += torch.sum(torch.eq(labels, torch.round(outputs)))
            all_labels.append(labels.cpu())
            all_outputs.append(outputs.cpu())
            print(f'outputs: {outputs}')
            print(f'labels: {labels}')
            print(f'error: {torch.sum(torch.sub(outputs, labels))/4}')
            print(f'errorabs: {torch.sum(torch.abs(torch.sub(outputs, labels)))/4}')
    print(f'MRE: {error/total:.5f}\nMARE: {errorabs/total:.5f}\naccuracy:{correct/total:.5f}')
    #print("accuracies: ")
    #print(accuracies)
    #print("MAREs: ")
    #print(mares)
    

    #mutation testing
    mutation_accuracy = 0
    if mutation_file != '':
        data = pd.read_csv(mutation_file)
        X = data.loc[:, features]
        y = data.loc[:, ["cat", "actual_cat"]]
        
        if type == "credit":
            norm_X = (X-og_dataset_min)/(og_dataset_max-og_dataset_min)
            norm_X['preffed_custommer'] = X["preffed_custommer"]
        elif type == "heart_risk":
            norm_X = (X-og_dataset_min)/(og_dataset_max-og_dataset_min)
            norm_X['gender'] = X["gender"]
            norm_X['smoking'] = X["smoking"]

        validation_data = Data(norm_X, y)
        validationLoader = DataLoader(validation_data, batch_size=1, shuffle=False, num_workers=2)
        correct_as_correct, correct_as_incorrect, incorrect_as_correct, incorrect_as_incorrect = 0, 0, 0, 0
        all_outputs_mutation, all_labels_mutation = [], []
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
                    all_outputs_mutation.append(1)
                    all_labels_mutation.append(1)
                #Correct records classified as Incorrect (Faulty)
                if outputs[0][0] != labels[0][1] and labels[0][0] == labels[0][1]:
                    correct_as_incorrect += 1
                    all_outputs_mutation.append(1)
                    all_labels_mutation.append(0)
                #Incorrect (Faulty) records classified as Correct
                if outputs[0][0] != labels[0][1] and outputs[0][0] == labels[0][0]:
                    incorrect_as_correct += 1
                    all_outputs_mutation.append(0)
                    all_labels_mutation.append(1)
                #Incorrect (Faulty) records classified as Incorrect (Faulty)
                if outputs[0][0] != labels[0][0] and labels[0][1] != labels[0][0]:
                    incorrect_as_incorrect += 1
                    all_outputs_mutation.append(0)
                    all_labels_mutation.append(0)
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

        if return_result == 0:
            plot_roc_curve(all_labels_mutation, all_outputs_mutation)

    #visualize
    if return_result == 1:
        if type == "triangle":
            return correct/total
        elif type == "credit":
            return mutation_accuracy
        elif type == "heart_risk":
             return mutation_accuracy
    else:
        plot_actual_predicted(all_labels, all_outputs)

def main(type, epochs, hidden_layer_size, leaning_rate):
    if __name__ == '__main__':
        freeze_support()
        train_rbf(type, epochs, hidden_layer_size, leaning_rate, 0)

#triangle
#main("triangle", 200, 200, 0.005)

#bank_credit
#main("credit", 200, 200, 0.005)

#heart_risk
main("heart_risk", 200, 200, 0.005)