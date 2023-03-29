import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import matplotlib.pyplot as plt
# Whatever other imports you need

# You can implement classes and helper functions here too.

class MLP(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size,args.hidden1)
        self.activation = 0
        if args.non_linearity ==1:
            self.activation = nn.ReLU()
        elif args.non_linearity ==2:
            self.activation = nn.Sigmoid()
        
        self.output_layer = nn.Linear(args.hidden1,output_size)
        self.logsoftmax =nn.LogSoftmax(dim=1)

    def forward(self, data):
        after_input_layer = self.input_layer(data)
        if self.activation:
            activation = self.activation(after_input_layer)
            hidden = self.output_layer(activation)
        else:
            hidden = self.output_layer(after_input_layer)
        
        output=self.logsoftmax(hidden)

        return output 

def model_trainning(train_set,pred_set,epochs,batch_size):
    train_set = torch.utils.data.TensorDataset(train_set,pred_set)
    train_set = [sample for sample in train_set if sample[1].numel() > 0]
    dataloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)

    model = MLP(input_size=200,output_size=14)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            model_input = batch[0]
            ground_truth = batch[1]
            output = model(model_input)
            loss = loss_fn(output, ground_truth.long())
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model
#model = model_trainning(train_vectors,train_labels,epochs=4,batch_size=10)
    

def confusion_matrix_printed(model,test_vectors,test_labels,neurons):
    model.eval()
    pred = model(test_vectors).argmax(dim=1)
    cm = confusion_matrix(test_labels,pred)
    sn.heatmap(cm, annot=True)
    plt.savefig(f"Confusion_Matrix_{neurons}.png")
#confusion_matrix_printed(model,test_vectors,test_labels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefiletrain", type=str, help="The file containing the table of instances and features training.")
    parser.add_argument("featurefiletest", type=str, help="The file containing the table of instances and features test.")
    parser.add_argument("hidden1",type=int)
    parser.add_argument("non_linearity",type=int,default=0)
    
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()
    train_table = pd.read_csv(args.featurefiletrain)
    test_table= pd.read_csv(args.featurefiletest)
    encoder = preprocessing.LabelEncoder()
    label_train = encoder.fit_transform(train_table.label.to_numpy().reshape(-1,1))
    label_test =encoder.transform(test_table.label.to_numpy().reshape(-1,1))
    train_vectors = torch.Tensor(train_table.iloc[:,:-1].to_numpy())
    train_labels = torch.LongTensor(np.array(label_train))
    test_vectors = torch.Tensor(test_table.iloc[:,:-1].to_numpy())
    test_labels = torch.LongTensor(np.array(label_test))
    model = model_trainning(train_vectors,train_labels,epochs=4,batch_size=10)
    confusion_matrix_printed(model,test_vectors,test_labels,args.hidden1)

    print("Reading {}&{}...".format(args.featurefiletest,args.featurefiletrain))

    # implement everything you need here
    
