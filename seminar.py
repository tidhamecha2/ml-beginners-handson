# sample=500 jbsub -cores 1+1 -q x86_24h -e joblogs/err$sample.txt -o joblogs/out$sample.txt -mem 10g python mnist.py $sample
# Easy implementation based on https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.optim import Optimizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys
from tqdm import tqdm

import pdb

transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                            transforms.Lambda(lambda x: torch.flatten(x))])


input_size = 784
hidden_sizes = [128, 64]
output_size = 10

batch_amp = 10
batch_size= 64 * batch_amp
steps_expected = 500 // batch_amp

# lr=torch.nn.Parameter(torch.FloatTensor([0.003])) * batch_amp
lr = 0.003 * batch_amp
momentum=0.9
# epochs = 200



samples_per_class = 100
fullset = datasets.MNIST('./data/mnist', train=True, transform=transform)
indices = np.arange(len(fullset))
train_indices, test_indices = train_test_split(indices, train_size=samples_per_class*10, stratify=fullset.targets)
train_dataset = Subset(fullset, train_indices)
test_dataset = Subset(fullset, test_indices)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_accuracy(datasetloader, model):
    correct_count, all_count = 0, 0
    preds = None
    out_label_ids = None
    
    for images,labels in datasetloader:
        with torch.no_grad():
            images = images.to(device)
            logps = model(images)

            if preds is None:
                preds = torch.exp(logps).detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:        
                preds = np.append(preds, torch.exp(logps).detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    acc = (preds == out_label_ids).mean()
    all_count = len(preds)
    return all_count, acc

def exp_kmeans():

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    train_x = next(iter(train_loader))[0].numpy()


    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_x)
    kmeans.cluster_centers_
    _, axs = plt.subplots(1,n_clusters)
        
    for i in range(n_clusters):
        pc = kmeans.cluster_centers_[i].reshape(28,28)
        axs[i].imshow(pc, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.savefig("clusters.jpeg")

def exp_pca():

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    train_x = next(iter(train_loader))[0].numpy()


    n_components = 10
    pca = PCA(n_components=n_components, random_state=0).fit(train_x)

    _, axs = plt.subplots(1,n_components)
        
    for i in range(n_components):
        pc = pca.components_[i].reshape(28,28)
        axs[i].imshow(pc, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.savefig("pcs.jpeg")


def exp_lr():

    from sklearn.linear_model import LogisticRegression

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    train= next(iter(train_loader))
    train_x = train[0].numpy()
    train_y = train[1].numpy()

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    test= next(iter(test_loader))
    test_x = test[0].numpy()
    test_y = test[1].numpy()

    print("Begin training LR")
    clf = LogisticRegression(random_state=0,solver='liblinear').fit(train_x, train_y)

    print("Train set accuracy:{}".format(clf.score(train_x, train_y)))
    print("Test set accuracy:{}".format(clf.score(test_x, test_y)))
    

def exp_nn():                    
    run = 0
    
    print("begin traiing NN")
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    n_batches = len(trainloader)
    epochs = steps_expected // n_batches
    print("total steps:{}".format(epochs * n_batches))
    print("batch size:{}\nepochs:{}\nlr:{}".format(batch_size,epochs,lr))
    

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),   
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))
    model.to(device)
    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    # images = images.view(images.shape[0], -1)
    images, labels = images.to(device), labels.to(device)
    logps = model(images) #log probabilities
    loss = criterion(logps, labels) #calculate the NLL loss

    # print('Before backward pass: \n', model[0].weight.grad)
    loss.backward()
    # print('After backward pass: \n', model[0].weight.grad)

    # optimizer = MySGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    time0 = time()
    step = 0

    
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            images, labels = images.to(device), labels.to(device)
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            step = step + 1
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
            # val_acc, val_count = 0, 0

            val_count, val_acc = model_accuracy(valloader, model)
            train_count, train_acc = model_accuracy(trainloader, model)

            print('{} {} {}'.format(step, val_acc, train_acc))
            
            


exp_pca()
exp_kmeans()
exp_lr()
exp_nn()



    