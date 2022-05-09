import pytorch_lightning as pl
from torchvision import transforms
import torch
import pandas as pd
import glob
from PIL import Image
import torch.cuda as cuda

from hyperparameters import parameters as params
from network import Net

def make_prediction(img_path): 
    pl.seed_everything(params['seed'])
    # Setup model
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = Net.load_from_checkpoint('../weights/Efficientnet_weights-epoch=00-v1.ckpt').to(device) #change '../weights/Efficientnet_weights-epoch=00-v1.ckpt' if dir is different
    model.eval()

    # Transforms
    in_transform = transforms.Compose([
                transforms.Resize((params['img_size'], params['img_size']), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(params['data_mean'], params['data_std'])])
    out_transform = torch.nn.Sigmoid()

    img = Image.open(img_path).convert('RGB')
    img = in_transform(img)
    img = img.type(torch.float32).unsqueeze(0).to(device)
    output = model(img)
    prob = out_transform(output).cpu().item()
    prediction = int(prob >= 0.5)
    
    return prediction 