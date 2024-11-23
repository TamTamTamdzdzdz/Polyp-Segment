import torch
from models import UNET, load_model
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision.transforms import PILToTensor, ToPILImage
import torch.nn.functional as F
import os
from train import device
import argparse


checkpoint_path = 'unet_model.pth'

model = UNET(out_channels=3).to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

transform = PILToTensor()
to_pil = ToPILImage()

def infer(img_path):
    data = Image.open(img_path)
    
    # Transform the image
    transform = PILToTensor()
    transform_data = transform(data) / 255.0  # Normalize to [0, 1]
    transform_data = transform_data.to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model(transform_data.unsqueeze(0))
        pred_tensor = F.one_hot(torch.argmax(prediction[0], 0).cpu()).float()
        pred_tensor = pred_tensor.cpu()  # Move back to CPU for processing
    
    # Visualize and save the result
    pred_img = ToPILImage()(pred_tensor.permute(2, 0, 1))  # Permute channels for PIL
    plt.imshow(pred_tensor)  # Adjust cmap as needed
    plt.show()
    
    # Save the result in 'output' directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  # Create 'output' directory if it doesn't exist
    output_file_name = f'segment_{os.path.basename(img_path)}'  # Only the image name
    output_path = os.path.join(output_dir, output_file_name)
    

    pred_img.save(output_path)
    print(f"Segmented image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    
    infer(args.image_path)
