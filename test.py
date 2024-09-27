from unet import UNet
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

model = torch.load('./models/unet.pth')
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((320, 320)), 
        transforms.ToTensor(),        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    
    image_tensor = preprocess(image).unsqueeze(0)  # type: ignore
    return image_tensor

def perform_inference(image_tensor):
    with torch.no_grad():  
        output = model(image_tensor)  
    
    alpha_matte = torch.sigmoid(output[0, 0]).cpu().numpy()  
    return alpha_matte

def visualize_result(alpha_matte):
    plt.imshow(alpha_matte, cmap='gray')
    plt.axis('off')
    plt.show()


image_path = './test1.jpg'
image_tensor = preprocess_image(image_path)
alpha_matte = perform_inference(image_tensor)
visualize_result(alpha_matte)
