from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn import tree


#The function will read files and seprate it into data and labes and returns the data
def read(filename):
    whole_file_data = open(filename,"r")
    whole_file_data_reader = csv.reader(whole_file_data, delimiter=",")
    data=[]
    lables=[]
    for row in whole_file_data_reader:
        data.append(row[0:-2])
        lables.append(row[-1])

    data =np.array(data).astype(np.float)
    lables=np.array(lables).astype(np.float)
    return data,lables


#The function execute Perceptron algorithm  which takes the data and lablescraeted from read function  along with the file name(for identification).Also the function is been passed with diffrent parameters for MLPClassifier.The data and labels are split into trainig and testing data. Weights  and threshold are calcluated using train data and the test data is used to predict the precision. This is done using sklearn MLPClassifier
def Perceptron(data,lables,hidden_layer_sizes,max_iter,solver,learning_rate,learning_rate_init):
    lenthof75=round(len(data)*.75)
    train_data=data[:lenthof75]
    train_labels=lables[:lenthof75]
    test_data=data[lenthof75:]
    test_labels=lables[lenthof75:]

    lr=0.01
    ws= (np.random.random(data.shape[1])/10)-0.03
    t=(np.random.random()/10)-0.03


    # Prediction Space with passed parametres .MLPClassifier from sklearn is used

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,solver=solver,learning_rate=learning_rate,learning_rate_init=learning_rate_init)
    clf.fit(train_data, train_labels)
    pred = clf.predict(test_data)
    correct=0
    for i,xs in enumerate(test_data):
        if(pred[i]==test_labels[i]):
            correct=correct+1
    print("MLPClassifier Accurecy"+" : "+str((correct/test_data.shape[0])*100)+"%")
    print(str(clf.n_iter_)+" itreation"+" hidden_layer_sizes : "+str(hidden_layer_sizes)+" max_iter : "+str(max_iter))
    print(" solver : "+str(solver)+" learning_rate : "+str(learning_rate)+" learning_rate_init : "+str(learning_rate_init))
      #  print(pred)





#DecisionTreeFor for comparing the MLPClassifier
def runtheDecisionTreeForTestingData(data,lables):
    lenthof75=round(len(data)*.75)
    train_data=data[:lenthof75]
    train_labels=lables[:lenthof75]
    test_data=data[lenthof75:]
    test_labels=lables[lenthof75:]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_labels)
    prediction = clf.predict(train_data)
    correct = (prediction == train_labels).sum()
    prediction = clf.predict(test_data)
    correct = (prediction == test_labels).sum()
    result1=(correct/len(prediction)*100)
    print("Decision Tree Accurecy: "+str(np.round(result1,3)))




#All the 4 provided files and a new file from UCI are tested using  Perceptron learning algorithm. Initially the file is passed yo read funtion and the returned data is passed to Perceptron function for the prediction. The new file from UCI is called Epileptic_Seizure_Recognition_Data_Set is a  very commonly used dataset featuring epileptic seizure detection.


print("------------------------")

#runtheDecisionTreeForTestingData(data1,lables1,"000759854_1.csv")
data1,lables1=read("000759854_1.csv")
print("File :"+"000759854_1.csv")
runtheDecisionTreeForTestingData(data1,lables1)

Perceptron(data1,lables1,(10,6,10),5000,'sgd','adaptive',0.0005)

print("------------------------")


data1,lables1=read("000759854_2.csv")
print("File :"+"000759854_2.csv")
runtheDecisionTreeForTestingData(data1,lables1)

Perceptron(data1,lables1,(10,6,10),5000,'sgd','adaptive',0.001)
print("------------------------")

data1,lables1=read("000759854_3.csv")
print("File :"+"000759854_3.csv")
runtheDecisionTreeForTestingData(data1,lables1)

Perceptron(data1,lables1,(16,8,8),5000,'sgd','adaptive',0.0002)
print("------------------------")

data1,lables1=read("000759854_4.csv")
print("File :"+"000759854_4.csv")
runtheDecisionTreeForTestingData(data1,lables1)

Perceptron(data1,lables1,(18,16,6),5000,'sgd','adaptive',0.0002)


print("------------------------")

data1,lables1=read("Epileptic_Seizure_Recognition_Data_Set.csv")
print("File :"+"Epileptic_Seizure_Recognition_Data_Set.csv")
runtheDecisionTreeForTestingData(data1,lables1)
print("calculating!!!!!!!!!!!!!!!")
Perceptron(data1,lables1,(25,16,20),5000,'sgd','adaptive',0.0002)



