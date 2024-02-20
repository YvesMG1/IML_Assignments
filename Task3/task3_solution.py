# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    pretrained = AutoModelForImageClassification.from_pretrained("Kaludi/food-category-classification-v2.0")
    model = nn.Sequential(*list(pretrained.children())[:-1])
    model.eval()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.clamp(0, 1)) 
    ])

    # Define the dataset and the dataloader
    BATCH_SIZE = 32
    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    pin_memory=True, num_workers=8)

    # Iterate over the images in the dataloader and generate embeddings for each batch
    embeddings = []
    for batch in train_loader:
        images, _ = batch 
        with torch.no_grad():
            model_output = model(images)
            output_tensor = model_output.last_hidden_state.detach()
            pooled_output = F.adaptive_avg_pool1d(output_tensor.transpose(1, 2), output_size=1).transpose(1, 2)
            batch_embeddings = pooled_output.squeeze()  

        embeddings.append(batch_embeddings.cpu().numpy())

    # Concatenate the embeddings from all batches into a single numpy array
    embeddings = np.concatenate(embeddings, axis=0)

    np.save('dataset/embeddings_kaludi_food.npy', embeddings)
    
def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings_kaludi_food.npy')
    
    mean_embedding = np.mean(embeddings, axis=0)
    std_embedding = np.std(embeddings, axis=0)
    normalized_embedding = (embeddings - mean_embedding) / std_embedding

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = normalized_embedding[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

#TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
#TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(3072, 512) #food embeddings
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output
    
# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 1):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

def train_model(train_loader, val_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 20

    f_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    if not os.path.exists('models'):
        os.makedirs('models')

    for epoch in range(n_epochs):
        model.train()
            
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            train_loss = f_loss(output.squeeze(), target.float())
            train_loss.backward()
            optimizer.step()
            
            if batch_idx % 1000 == 0:
                print('Epoch {}, Batch idx {}, training loss {}'.format(
                    epoch, batch_idx, train_loss.item()))
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                output = model(data)
                loss = f_loss(output.squeeze(), target.float())
                val_loss += loss.item() * target.size(0)
        
        val_loss /= len(val_loader.dataset)
            
        print('Epoch {}, validation loss {}'.format(epoch, val_loss))
        model_path = os.path.join('models', f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        scheduler.step()
        return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)
    
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    indices = np.random.permutation(len(X))
    val_size = int(len(X) * 0.2)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = X[train_indices,:]
    y_train = y[train_indices]
    X_val = X[val_indices,:]
    y_val = y[val_indices]

    train_loader = create_loader_from_np(X_train, y_train, train = True, batch_size=32)
    val_loader = create_loader_from_np(X_val, y_val, train = True, batch_size=32)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader, val_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
