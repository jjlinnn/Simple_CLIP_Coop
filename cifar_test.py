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
from torch import nn
import torchvision
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from util import AvgMeter,get_lr
from torchvision.datasets import CIFAR10
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np
from models import *
from dataset import *


if __name__ == '__main__':
    model = CLIPModel()
    model.load_state_dict(torch.load('best.pt'))
    model.to(CFG.device)
    model.eval()

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
    with torch.no_grad():
        text_features = text_encoder(input_ids, attention_mask)#torch.Size([10, 768])

    with torch.no_grad():
        text_embeddings = model.text_projection(text_features)#torch.Size([10, 256])
    correct = 0
    for image, label in cifar10_data:
        image_input = image.unsqueeze(0).to(CFG.device)
        image_features = model.image_encoder(image_input)
        image_embeddings = model.image_projection(image_features)
        similarity = torch.matmul(image_embeddings, text_embeddings.T).squeeze(0)

        predicted_class = similarity.argmax().item()
        # 21.39
        # if predicted_class == label: # 23.73%;37.69%;51.62%
        #     correct += 1

        _, topk_predicted_classes = similarity.topk(1)
        if label in topk_predicted_classes:
            correct += 1

    accuracy = correct / len(cifar10_data)
    print(f"Zero-shot classification accuracy: {accuracy * 100:.2f}%")