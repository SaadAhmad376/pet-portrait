import torch
import torchvision
from torch.utils.data import random_split
from torchvision import transforms
import os
import matplotlib.pyplot as plt


from networks.vae import VAE
from losses import psnr_loss, pixel_to_pixel_loss




# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


dataset = torchvision.datasets.ImageFolder(root='path_to_dataset', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])





# Initialize the model
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Define the VAE loss here
    pass

num_epochs = 50  # Define the number of epochs

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Save model and results every 10 epochs
    if epoch % 10 == 0:
        os.makedirs(f'results/epoch_{epoch}', exist_ok=True)
        torch.save(model.state_dict(), f'results/epoch_{epoch}/model.pth')

        # Save some sample output images
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            for i in range(5):  # Save 5 sample images
                plt.imshow(sample[i][0], cmap='gray')
                plt.savefig(f'results/epoch_{epoch}/image_{i}.png')

