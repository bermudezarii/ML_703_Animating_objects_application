
import pytorch_lightning as pl
from torchvision import transforms
import torch
import pandas as pd
import glob
from PIL import Image

from hyperparameters import parameters as params
from network import Net


pl.seed_everything(params['seed'])

# Setup model
device = 'cuda' if cuda.is_available() else 'cpu'
model = Net.load_from_checkpoint('../weights/CHANGE TO TRAINED WEIGHTS').to(device)
model.eval()

# Transforms
in_transform = transforms.Compose([
            transforms.Resize((params['img_size'], params['img_size']), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(params['data_mean'], params['data_std'])])
out_transform = torch.nn.Sigmoid()

# Data
test_data = glob.glob('PATH TO FOLDER WITH IMAGES')
rows = []

# Prediction
with torch.no_grad():

    for idx, filename in enumerate(test_data):
        print(idx)

        img = Image.open(filename).convert('RGB')
        img = in_transform(img)
        img = img.type(torch.float32).unsqueeze(0).to(device)
        output = model(img)
        prob = out_transform(output).cpu().item()
        prediction = int(prob >= 0.5)

        rows.append({'ID_IMG':filename, 'PREDICTION':prediction})


# Save predictions
df = pd.DataFrame(rows, columns=['ID_IMG','PREDICTION'])
df.to_csv('../inference_results.csv')
