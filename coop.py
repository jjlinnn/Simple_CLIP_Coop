import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from config import CFG as CFG
import torch
from tqdm import tqdm
from torch import nn
import torchvision
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from util import AvgMeter,get_lr
from torchvision.datasets import CIFAR10
import torch
from util import AvgMeter
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np
import torch.optim as optim
from models import *
from dataset import *
from config import CFG



def evaluate_zero_shot(model, text_embeddings, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.image_encoder(images)
            image_embeddings = model.image_projection(image_features)

            similarity = torch.matmul(image_embeddings, text_embeddings.T)
            predictions = similarity.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

class ExtendedTextEncoder(TextEncoder):
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, 1:5, :]


if __name__ == '__main__':
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load('best.pt'))
    model.to(CFG.device)
    model.eval()
    extended_text_encoder = ExtendedTextEncoder().to(CFG.device)
    original_text_encoder = model.text_encoder

    cifar10_labels = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]
    cifar10_data = CIFAR10(root=".", train=False, transform=Compose([Resize((224,224)), ToTensor()]),download=True)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    text_encoder = TextEncoder().to(CFG.device)
    text_inputs = [f"a photo of {c}" for c in cifar10_labels]
    encoded_inputs = tokenizer(text_inputs, padding='max_length', max_length=100, truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(CFG.device)#[10,100]
    attention_mask = encoded_inputs['attention_mask'].to(CFG.device)

    init_hiddenstate = extended_text_encoder(input_ids, attention_mask).to(CFG.device)#(4,768)

    trainable_hidden_state = nn.Parameter(init_hiddenstate[:, 0:3, :].clone(), requires_grad=True)
    fixed_hidden_state = init_hiddenstate[:, 3, :].detach()

    dataloader = torch.utils.data.DataLoader(cifar10_data, batch_size=64, shuffle=True)
    optimizer = optim.SGD([trainable_hidden_state], lr=0.01)
    accuracy = evaluate_zero_shot(model, model.text_projection(torch.cat([trainable_hidden_state, fixed_hidden_state.unsqueeze(1)], dim=1).mean(dim=1)), dataloader, CFG.device)

    print( 'Zero-shot Accuracy:', accuracy)# 23.75%
    #Epoch: 0 Zero-shot Accuracy: 0.2884
    #Epoch: 1 Zero-shot Accuracy: 0.3135
    #Epoch: 2 Zero-shot Accuracy: 0.3292
    #Epoch: 3 Zero-shot Accuracy: 0.3423

    for epoch in range(50):
        model.train()
        avg_loss = AvgMeter()
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{50}')
        for images, labels in progress_bar:
            images = images.to(CFG.device)
            labels = labels.to(CFG.device)

            image_features = model.image_encoder(images)
            image_embeddings = model.image_projection(image_features)

            full_hidden_state = torch.cat([trainable_hidden_state, fixed_hidden_state.unsqueeze(1)], dim=1)
            full_text_embeddings = model.text_projection(full_hidden_state.mean(dim=1))

            similarity = torch.matmul(image_embeddings, full_text_embeddings.T)
            loss = nn.CrossEntropyLoss()(similarity, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss.update(loss.item())
            progress_bar.set_postfix({'loss': avg_loss.avg})

        progress_bar.close()
        print('Epoch:', epoch, 'Loss:', avg_loss.avg)

        full_hidden_state = torch.cat([trainable_hidden_state, fixed_hidden_state.unsqueeze(1)], dim=1)
        full_text_embeddings = model.text_projection(full_hidden_state.mean(dim=1))
        accuracy = evaluate_zero_shot(model, full_text_embeddings, dataloader, CFG.device)

        print('Epoch:', epoch, 'Zero-shot Accuracy:', accuracy)