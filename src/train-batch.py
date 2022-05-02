from config import EMBEDDING_PATH, SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH,SAMPLE_TEST_DATA_PATH,SAMPLE_TEST_EDGE_PATH
from data import DataLoader
from model import HeterSumGraph
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def save_model(model, save_file):
        with open(save_file, 'wb') as f:
            torch.save(model.state_dict(), f)

if __name__ == "__main__":

    dw = 50
    ds = 384
    dh = 64
    de = 64
    heads = 1

    data_loader = DataLoader(SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH)
    print("created data loader")

    model = HeterSumGraph(dw, ds, dh, de, heads)
    print("created model")

    model.to(device)


    criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = torch.nn.BCELoss()
    #criterion = torch.nn.CrossEntropyLoss()


    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


    train_loader = data_loader
    test_loader = DataLoader(SAMPLE_TEST_DATA_PATH,SAMPLE_TEST_EDGE_PATH,EMBEDDING_PATH)

    EPOCHS = 5

    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    model.train()
    for epoch in range(EPOCHS):
        model.train()
        train_epoch_loss = 0.0
        train_examples = 0
        train_loss = 0.0
        test_loss = 0.0
        test_epoch_loss = 0.0
        test_examples = 0

        # Training
        correct_train = 0
        correct_test = 0

        all_test_preds= []
        all_test_y = []

        cnt = 0

        for (Xw, Xs, E, Erev), y in train_loader:
            print("Training example")
            #print(cnt)
            cnt+=1

            
            optimizer.zero_grad()

            preds = model.forward(Xw, Xs, E, Erev)
            print("Sigmoid scores")
            print(torch.nn.Sigmoid()(preds))
            #print(preds)

            # For GPU
            Xw = Xw.to(device,dtype = torch.long)
            Xs =  Xs.to(device,dtype = torch.long)
            E  =  E.to(device,dtype = torch.long)
            Erev = Erev.to(device,dtype = torch.long)

            #Accuracy
            n = preds.squeeze(1).detach().numpy().tolist()
            print("Logit predictions")
            print(n)
            p = [1  if item > 0 else 0 for item in n]
            a = y.numpy().tolist()
            print("Predicted classes")
            print(p)
            print("actual classes")
            print(a)
            for i,v in enumerate(a):
                if p[i]==v:
                    correct_train +=1

            label = y
            #print(y)
            # compute loss
            loss  = criterion(preds, label.float().unsqueeze(1))
            # print(preds.shape)
            # print(criterion(preds, label))
            # print(criterion(preds, label).unsqueeze(-1))
            # loss  = criterion(preds, label).unsqueeze(-1)

            print(loss)
           
            
            loss.backward() 
            optimizer.step()    

            # loss of the batch
            train_loss += float(loss.item())*y.shape[0]
            train_examples += y.shape[0]
            #print(train_loss)





        #Evaluation
        print("Evaluation")
        model.eval() # prep model for evaluation
        cnt = 0

        with torch.no_grad():
            for (Xw, Xs, E, Erev), y in test_loader:
                #print(cnt)
                cnt+=1
                preds = model.forward(Xw, Xs, E, Erev)
                label = y

                # Accuracy
                n = preds.squeeze(1).detach().numpy().tolist()
                p = [1 if item > 0 else 0 for item in n]
                a = y.numpy().tolist()
                print("Logit predictions")
                print(n)
                print("Predicted classes")
                print(p)
                print("actual classes")
                print(a)
                for i,v in enumerate(a):
                    if p[i]==v:
                        correct_test +=1

                loss  = criterion(preds, label.float().unsqueeze(1))
                #print(loss)

                all_test_preds = np.append(all_test_preds,p)
                all_test_y = np.append(all_test_y,np.array(y))


                # loss of the batch

                test_loss += float(loss.item())*y.shape[0]

                #print(len(y))
                test_examples += len(y)
                #print(test_loss)


        epoch_list.append(epoch)

        # Average loss over an epoch
        #print(len(train_loader))
        train_epoch_loss = train_loss / train_examples
        print("Epochs train loss {} ".format( train_epoch_loss))
        train_loss_list.append(train_epoch_loss)


        test_epoch_loss = test_loss / test_examples
        print("Epochs test loss {}".format(test_epoch_loss))
        test_loss_list.append(test_epoch_loss)

         # Training accuracy at epochs
        train_acc = correct_train/train_examples
        print("Train Accuracy {}".format(train_acc))
        train_acc_list.append(train_acc)

        # Test accuracy at epochs
        print(test_examples)
        print(correct_test)
        test_acc = correct_test/test_examples
        print("Test Accuracy {}".format(test_acc))
        test_acc_list.append(test_acc)



    path = '../models/ext_model_' + str(EPOCHS) +  'e.pth'
    save_model(model, path)


    df_results = pd.DataFrame(list(zip(train_loss_list,test_loss_list,train_acc_list,test_acc_list)),columns=['Train Loss','Test Loss','Train Accuracy','Test Accuracy'])
    path1 = '../results/report' + str(EPOCHS) +  'e.csv'
    df_results.to_csv(path1,index=None,sep=',')


    class_report = pd.DataFrame(classification_report(all_test_y, all_test_preds,output_dict=True)).transpose()
    path2 = '../results/classification_report' + str(EPOCHS) +  'e.csv'
    class_report.to_csv(path2)

    conf_matrix = pd.DataFrame(confusion_matrix(all_test_y,all_test_preds))
    #print(conf_matrix)
    sns.set(font_scale=1.4) # for label size
    sns_plot = sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 8}).get_figure() # font size
    path5 = '../results/GNN_confusion_matrix' + str(EPOCHS) +  'e.png'
    sns_plot.savefig(path5)

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(epoch_list, train_loss_list)
    plt.plot(epoch_list,test_loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['Train Loss','Test Loss'])

    path3 = '../results/GNN_loss' + str(EPOCHS) +  'e.png'
    plt.savefig(path3)


    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    plt.plot(epoch_list, train_acc_list)
    plt.plot(epoch_list,test_acc_list)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['Train Accuracy','Test Accuracy'])
    path4 = '../results/GNN_accuracy' + str(EPOCHS) +  'e.png'
    plt.savefig(path4)


