from PIL import Image
file_path = ""

image = Image.open(file_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
input_image = transform(image).unsqueeze(0)

# Load the trained model
model = torchvision.models.efficientnet_b4(pretrained=False)  
num_classes = len(train_dataset.classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.load_state_dict(torch.load('models/final_model.pth')) 
model.to(device)
model.eval()

with torch.no_grad():
    input_image = input_image.to(device)
    output = model(input_image)
    _, predicted = torch.max(output, 1)

predicted_class = train_dataset.classes[predicted.item()]
print("Predicted Face Shape:", predicted_class)
