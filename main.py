# This is a sample Python script.


# import the necessary packages
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch

import os
import pyttsx3


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    engine = pyttsx3.init()
    voice = engine.getProperty('voices')
    for voices in voice:
        print(voices)
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
            dataset = datasets.ImageFolder("venv/test", transform=transform
                                           )

            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )

            return data_loader

        # load the dataset



        valid_dataset = datasets.ImageFolder("venv/test", transform=transform)


        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size)

        return valid_loader

    #file = open("venv/label.txt", 'r')
    #classes = []
    #for i in file:
    #    classes.append(i.strip())
    classes = os.listdir("venv/animals")
    classes.sort()
    # CIFAR10 dataset
    valid_loader = data_loader(batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load("nouse.pth").to(device)

    model.eval()
    class_name = []
    with torch.no_grad():
        for _, (image, label) in enumerate(valid_loader):

            image = image.to(device)




            output = model(image)
            _, predicted = torch.max(output.data, 1)
            print(predicted)

            # display the result in terminal and show the input image
            print("[INFO]predicted label: {}".format(classes[predicted.item()]))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
