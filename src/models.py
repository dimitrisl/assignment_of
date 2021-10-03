import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

torch.manual_seed(42)


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
    elif ev_tr == "eval":
        model.eval()
    else:
        raise ValueError("NOT valid mode")
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
        loss = loss_function(outputs, labels.float())
        if ev_tr == "train":
            loss.backward()

            # 5 - update weights
            optimizer.step()

        loss_score.append(loss.detach().item())
        outputs = outputs.detach().numpy()
        tmp_out = np.zeros_like(outputs)
        outputs = torch.softmax(torch.FloatTensor(outputs), dim=1).numpy()
        tmp_out[np.arange(len(outputs)), outputs.argmax(axis=1)] = 1
        metric_score.append(f1_score(tmp_out.argmax(axis=1), labels.argmax(axis=1), average="macro"))
    # print("epoch {}: {} loss {} score {}".format(_epoch, ev_tr, np.average(loss_score), np.average(metric_score)))
    return np.average(loss_score), np.average(metric_score)


class CNNClassifier(nn.Module):

    def __init__(self, kernel_dim=100, kernel_sizes=(30, 40, 50), output_size=2):
        super(CNNClassifier, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(1, kernel_dim, (K, 1)) for K in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def forward(self, x):
        inputs = x.unsqueeze(1).unsqueeze(-1)
        inputs = [torch.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        concated = torch.cat(inputs, 1)
        out = self.fc(concated)
        return out


def main_cnn(data):
    BATCH = 2000
    raw_dataset, corresponding_labels = data
    oe_style = OneHotEncoder()
    corresponding_labels = oe_style.fit_transform(corresponding_labels.reshape(-1, 1))
    corresponding_labels = torch.FloatTensor(corresponding_labels.toarray())
    X_train, X_test, y_train, y_test = train_test_split(raw_dataset, corresponding_labels, test_size=0.2,
                                                        random_state=42)  # same machine same seed.
    vectorizer = TfidfVectorizer(analyzer='word')

    X_train = vectorizer.fit_transform(X_train.iloc[:, 0])
    X_test = vectorizer.transform(X_test.iloc[:, 0])


    X_train = csr_matrix(X_train)
    svd = TruncatedSVD(n_components=300, n_iter=7, random_state=42)
    X_train = svd.fit_transform(X_train)
    X_test = csr_matrix(X_test)
    X_test = svd.transform(X_test)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    train_dataset = SampleDataset(X_train, y_train)
    eval_dataset = SampleDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    criterion = torch.nn.BCEWithLogitsLoss()
    EPOCHS = 200
    lr = 0.001
    model = CNNClassifier()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    train_losses = []
    eval_losses = []

    train_scores = []
    eval_scores = []

    for i in range(EPOCHS):
        train_loss, train_score = train_epoch(i, train_loader, model, criterion, optimizer, "train")
        eval_loss, eval_score = train_epoch(i, eval_loader, model, criterion, optimizer, "eval")

        train_losses.append(train_loss)
        train_scores.append(train_score)

        eval_losses.append(eval_loss)
        eval_scores.append(eval_score)

    print("best evaluation scores are on {} {}".format(max(eval_scores), np.argmax(eval_scores)))
    print("best model loss score is {} on {} epoch".format(min(eval_losses), np.argmin(eval_losses)))
    return train_scores, eval_scores, train_losses, eval_losses
