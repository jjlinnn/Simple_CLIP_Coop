import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from config import CFG


class ImageEncoder(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=False, trainable=CFG.trainable):
        super().__init__()
        self.model = nn.Sequential()
        self.basemodel = models.resnet50()
        state_dict = torch.load("resnet50-11ad3fa6.pth")
        self.basemodel.load_state_dict(state_dict)
        self.basemodel = nn.Sequential(*list(self.basemodel.children())[:-1])
        for p in self.basemodel.parameters():
            p.requires_grad = trainable
        self.idi = torch.nn.Identity()

    def forward(self, x):
        x = self.basemodel(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.idi(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=False, trainable=CFG.trainable):
        super().__init__()

        self.model = DistilBertModel.from_pretrained(CFG.text_encoder_model)
        for p in self.model.parameters():
            p.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state#(10, 100, 768)
        #return last_hidden_state[:, self.target_token_idx, :]
        return last_hidden_state[:, 1:4, :].mean(dim=1)

class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()