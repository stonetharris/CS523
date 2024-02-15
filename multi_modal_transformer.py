import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import BertTokenizer, BertModel

class MultiModalTransformer(nn.Module):
    def __init__(self, text_model_name, image_model_fn, output_dim):
        super(MultiModalTransformer, self).__init__()

        self.text_model = BertModel.from_pretrained(text_model_name)
        self.image_model = image_model_fn(pretrained=True) 

        self.image_model.fc = nn.Identity()

        self.classifier = nn.Linear(self.text_model.config.hidden_size + self.image_model.fc.in_features, output_dim)

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[1] 
        
        image_features = self.image_model(images)

        combined_features = torch.cat((text_features, image_features), dim=1)

        logits = self.classifier(combined_features)

        return logits

model = MultiModalTransformer(
    text_model_name='bert-base-uncased',
    image_model_fn=resnet50,
    output_dim=10 
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = ["a photo of a cat"]
inputs = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
images = torch.rand(1, 3, 224, 224) 

logits = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
