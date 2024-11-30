# Gerekli kütüphaneleri yükleyin
import matplotlib.pyplot as plt
import numpy as np
import struct
from array import array
import os
from os.path import join
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
import torch.optim as optim

# MNIST veri setini yükleyen sınıf
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())   
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

# Veri yolları
input_path = '../input/mnist-dataset'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# MNIST veri yükleyicisini başlatın
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Görselleri ve başlıkları görselleştirme
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)        
        index += 1

images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)

# CNN Modeli Tanımlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_1 = nn.Sequential(
    nn.Conv2d(1, 28, 5, stride=1),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(28, 26, 5, stride=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(26 * 8 * 8, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

model = model_1
model.to(device)

# Eğitim verilerini tensor haline getirme
x_train_tensor = torch.tensor(np.array(x_train), dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Kayıp fonksiyonu ve optimizasyon
criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
epochs = 100

# Eğitim döngüsü
for e in range(epochs):
    model.train()
    sum_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y_hat = model(inputs)
        loss = criterion(y_hat, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    
    avg_loss = sum_loss / len(train_loader.dataset)
    print(f"epoch #{e + 1}")
    print(f"loss: {avg_loss}")

# Modeli kaydetme
torch.save(model.state_dict(), "model.pt")

# Modeli yükleyip değerlendirme
model.load_state_dict(torch.load("model.pt"))
model.eval()

x_test_tensor = torch.tensor(np.array(x_test), dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
