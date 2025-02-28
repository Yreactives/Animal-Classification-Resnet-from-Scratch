from model import ResidualBlock, ResNet, CNN
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_loader(
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    # define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        dataset = datasets.ImageFolder("venv/animals", transform = transform
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.ImageFolder("venv/animals", transform= transform
        )

    valid_dataset = datasets.ImageFolder("venv/animals", transform= transform
        )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=2)

    return (train_loader, valid_loader)

if __name__ == "__main__":
    # CIFAR10 dataset
    train_loader, valid_loader = data_loader(
                                             batch_size=64)

    test_loader = data_loader(
                              batch_size=64,
                              test=True)
    class_names = os.listdir("venv/animals")
    class_names.sort()
    #data = torch.load("data.pth")
    num_classes = len(class_names)
    num_epochs = 10

    #epochs = data['epochs']
    batch_size = 16
    learning_rate = 2e-4

    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes).to(device)
    #model = CNN(64, 124).to(device)
    #model.load_state_dict(data['model'])
    #model = torch.load("nouse.pth").to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)
    #optimizer.load_state_dict(data['optimizer'])
    #lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
    # Train the model
    total_step = len(train_loader)

    import gc



    for epoch in range(num_epochs):
        model.train()
        optimizer.param_groups[0]['lr'] *= 0.95991
        for i, (images, labels) in enumerate(train_loader):

            #optimizer.param_groups[0]['lr'] *= 0.99999
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, loss.item()))

        # Validation
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * correct / total))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        #"epochs": epochs + num_epochs
        "epochs": num_epochs
    }
    torch.save(data, "data2.pth")
    #traced_script_module = torch.jit.trace(model, )

    #with open("classes.txt", 'w') as fp:
        #for item in class_names:
            #fp.write(item + "\n")

    #torch.jit.trace()
    print("finished")
