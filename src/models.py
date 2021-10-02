import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder


class SampleDataset(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.labels = y

    def __getitem__(self, index):
        sample = torch.FloatTensor(self.data[index])
        labels = self.labels[index]
        return sample, labels

    def __len__(self):
        return len(self.data)


def train_epoch(_epoch, dataloader, model, loss_function, optimizer, ev_tr):
    # switch to train mode -> enable regularization layers, such as Dropout
    if ev_tr == "train":
        model.train()
    else:
        model.eval()
    loss_score = []
    metric_score = []

    for sample_batched in dataloader:
        # get the inputs (batch)
        inputs, labels = sample_batched

        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass: compute predicted y by passing x to the model
        outputs = model(inputs)
        # 3 - compute loss
        print(outputs)
        print(labels)
        loss = loss_function(outputs, labels.float())
        loss.backward()

        # 5 - update weights
        optimizer.step()

        loss_score.append(loss.detach().item())
        metric_score.append(f1_score(outputs, labels))
    print(ev_tr)
    print("loss", np.average(loss_score))
    print("score", np.average(metric_score))
    return np.average(loss_score), np.average(metric_score)


class CNNClassifier(nn.Module):

    def __init__(self, embeddings, kernel_dim=100, kernel_sizes=(3, 4, 5), output_size=2):
        super(CNNClassifier, self).__init__()
        # input
        embedding_dim = embeddings.shape[1]
        #  end of inputs.
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, 1)) for K in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def forward(self, x):
        inputs = x.unsqueeze(0).unsqueeze(1)
        inputs = [torch.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs = [torch.max_pool2d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        concated = torch.cat(inputs, 1)
        print(concated.shape)
        out = self.fc(concated)
        print(out.shape)
        return out


def cnn_pipeline(X_train, X_test, y_train, y_test):
    BATCH = 50
    oe_style = OneHotEncoder()
    y_train = torch.FloatTensor(oe_style.fit_transform(y_train.reshape(-1, 1)).getnnz())
    y_test = torch.FloatTensor(oe_style.fit_transform(y_test.reshape(-1, 1)).getnnz())
    train_dataset = SampleDataset(X_train, y_train)
    eval_dataset = SampleDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    criterion = torch.nn.BCEWithLogitsLoss()
    EPOCHS = 10
    lr = 0.001

    model = CNNClassifier(X_train)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    train_losses = []
    eval_losses = []

    train_scores = []
    eval_scores = []

    for i in range(EPOCHS):
        print("epoch: %s" % i)
        train_loss, train_score = train_epoch(i, train_loader, model, criterion, optimizer, "train")
        print("\n")
        eval_loss, eval_score = train_epoch(i, eval_loader, model, criterion, optimizer, "eval")

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        train_scores.append(train_score)
        eval_scores.append(eval_score)

    return train_scores, eval_scores, train_losses, eval_losses


import pickle

with open("lala.pkl", "rb") as lala:
    data = pickle.load(lala)


(X_train, X_test, y_train, y_test) = data
cnn_pipeline(X_train, X_test, y_train, y_test)