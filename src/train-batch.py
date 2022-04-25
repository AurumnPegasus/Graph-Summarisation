from config import EMBEDDING_PATH, SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH,SAMPLE_TEST_DATA_PATH,SAMPLE_TEST_EDGE_PATH
from data import DataLoader
from model import HeterSumGraph
import torch


dw = 50
ds = 384
dh = 64
de = 64
heads = 1

data_loader = DataLoader(SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH)
print("created data loader")

model = HeterSumGraph(dw, ds, dh, de, heads)
print("created model")

def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)

# Training

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()

train_loader = data_loader

EPOCHS = 10

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    train_loss = 0.0
    total_examples = 0

    for (Xw, Xs, E, Erev), y in train_loader:
        preds = model.forward(Xw, Xs, E, Erev)
        print(preds)
        print(y)
        #print(preds)
        label = y

        # compute loss
        loss  = criterion(preds, label.float().unsqueeze(1))
        print(loss)

        optimizer.zero_grad()

        # loss of the batch
        train_loss += float(loss.item())*y.shape[0]
        total_examples += y.shape[0]
        print(train_loss)

        loss.backward() 
        optimizer.step()    

    # Average loss over an epoch
    #print(len(train_loader))
    train_loss = train_loss / total_examples
    print("Epochs loss {}".format(train_loss))

path = '../models/ext_model_' + str(EPOCHS) +  'e.pth'
save_model(model, path)

model.load_state_dict(torch.load(path))
model.eval()
print(model)


## Evaluation

test_loader = DataLoader(SAMPLE_TEST_DATA_PATH,SAMPLE_TEST_EDGE_PATH,EMBEDDING_PATH)
with torch.no_grad():

    epoch_loss = 0.0
    test_loss = 0.0
    total_examples = 0

    for (Xw, Xs, E, Erev), y in test_loader:
        preds = model.forward(Xw, Xs, E, Erev)
        label = y

        loss  = criterion(preds, label.float().unsqueeze(1))
        print(loss)

        # loss of the batch
        test_loss += float(loss.item())*y.shape[0]
        total_examples += y.shape[0]
        print(test_loss)
    
    test_loss = test_loss / total_examples
    print("Epochs test loss {}".format(test_loss))
    
