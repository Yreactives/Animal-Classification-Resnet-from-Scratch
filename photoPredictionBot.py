import requests.exceptions
import telebot
import torch
import os
from torchvision import transforms
from PIL import Image
import wikipediaapi
import pyttsx3
from playsound import playsound
from gtts import gTTS
from io import BytesIO
import pygame
import wikipedia
from model import ResNet, ResidualBlock
from translate import Translator

bot = telebot.TeleBot(open('telegrambot.txt', 'r').read())

@bot.message_handler(content_types=['photo'])
def photoClassifier(message):
    device = torch.device("cpu")
    classes = os.listdir("venv/animals")
    classes.sort()
    data = torch.load("data.pth")
    model = ResNet(ResidualBlock, [3, 4, 6, 3], len(classes)).to(device)

    #model = torch.load("nouse.pth").to(device)
    model.load_state_dict(data['model'])
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    filepath = "venv/downloaded/" + file_info.file_path
    downloaded = bot.download_file(file_info.file_path)
    with open(filepath, "wb") as new_file:
        new_file.write(downloaded)
    file = Image.open(filepath)

    #convert = transforms.ToTensor()
   # resize = transforms.Resize((224, 224))
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    file = transform(file)

    file = torch.unsqueeze(file, 0)



    model.eval()

    with torch.no_grad():
        file = file.to(device)
        output = model(file)

        _, predicted = torch.max(output.data, 1)
        probability = torch.max(output.data).item()
        print(probability)
    if probability >= 3:

        bot.reply_to(message, "this is an image of " + classes[predicted.item()].capitalize())
        #wiki_wiki = wikipediaapi.Wikipedia("en")
        #ny = wiki_wiki.page(classes[predicted.item()].capitalize())

        translator = Translator(to_lang="id")
        mp3_fp = BytesIO()
        wikipedia.set_lang("id")
        sentences = ""
        sentences_count = 0
        max_count = 3
        sentences = wikipedia.summary(translator.translate(classes[predicted.item()]), sentences=3)




        tts = gTTS(sentences, lang="id")
        tts.write_to_fp(mp3_fp)
        pygame.init()
        pygame.mixer.init()
        mp3_fp.seek(0)
        pygame.mixer.music.load(mp3_fp, "mp3")
        #pygame.mixer.music.play()

        bot.reply_to(message, sentences)

    else:
        bot.reply_to(message, "I don't know what is this. does the photo have animal in?")


@bot.message_handler(commands=['start'])
def reply(message):
    bot.reply_to(message, "Hi I'm a bot that can classify images of animals like pokedex, try to send image and I will guess what is the animal present in that image. ")

@bot.message_handler(content_types=['audio', 'document'])
def reply(message):
    bot.reply_to(message, "please send an image so I can classify it")

@bot.message_handler(func=lambda message:True)
def reply(message):
    bot.reply_to(message, "please send an image so I can classify it")

bot.infinity_polling()

