import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
from PIL import Image

def train_vision_transformer(dataset_path, num_epochs=10, batch_size=32, learning_rate=1e-4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = vit_b_16(pretrained=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'vision_transformer.pth')
    return model

def predict_masked_pixels(model, image_path):
    image = Image.open(image_path).convert('RGB')  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0) 

    model.eval()  
    with torch.no_grad():
        predictions = model(image)

    predicted_pixels = predictions  

    return predicted_pixels

def fine_tune_vision_transformer(dataset_path, model_path, num_classes, num_epochs=10, batch_size=32, learning_rate=1e-4):
    model = vit_b_16(pretrained=False)  
    model.load_state_dict(torch.load(model_path))
    
    model.heads = nn.Linear(model.heads.in_features, num_classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'fine_tuned_vision_transformer.pth')