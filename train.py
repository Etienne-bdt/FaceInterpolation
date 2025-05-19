import torch
import torch.nn as nn
from torchvision.datasets import CelebA
from model import VAE
from loss import loss_function
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision import transforms as tvf

def train_vae(model, device, train_loader, val_loader, optimizer, gamma, num_epochs=10):
    model.train()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        for _, (data, _) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            mse, kld = loss_function(recon_batch, data, mu, logvar, gamma)
            loss = mse + kld
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()/len(train_loader)}')

        for _, (data, _) in enumerate(val_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            mse, kld = loss_function(recon_batch, data, mu, logvar, gamma)
            loss = mse + kld
            
        print(f"Validation Loss: {loss.item() / len(val_loader)}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with loss: {best_loss}")

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for data, _ in val_loader:
                recon_batch, mu, logvar = model(data)
                mse, kld = loss_function(recon_batch, data, mu, logvar, gamma)
                val_loss += mse.item() + kld.item()
            print(f'Validation Loss: {val_loss / len(val_loader)}')




def main():
    batch_size = 32
    dataset = CelebA(root='data', split='train', download=True, transform=tvf.Compose([tvf.ToTensor(),tvf.Resize((64, 64))]))
    len_train = int(len(dataset) * 0.8)
    len_val = len(dataset) - len_train
    train_set, val_set = torch.utils.data.random_split(dataset, [len_train, len_val])
    train_loader, val_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True), torch.utils.data.DataLoader(val_set, batch_size, shuffle=False)

    model = VAE(input_dim=3, hidden_dim=32, latent_dim=12288)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = torch.Tensor([0.1]).to(device)  # Example value for gamma
    gamma.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.add_param_group({'params': [gamma]})
    model = model.to(device)
    next(iter(train_loader))
    train_vae(model,device, train_loader, val_loader, optimizer, gamma, num_epochs=10)

    data,_ = next(iter(val_loader))
    data = data.to(device)
    recon_batch, mu, logvar = model(data)
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(data[i].cpu().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.subplot(2, 10, i + 11)
        plt.imshow(recon_batch[i].cpu().permute(1, 2, 0).detach().numpy())
        plt.axis('off')

    plt.savefig('reconstructions.png')
    plt.close()

    im1 = data[0]
    im2 = data[1]

    interpolated = model.interpolate(im1, im2, num_steps=10)
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(1, 11, i + 1)
        plt.imshow(interpolated[i].cpu().permute(1, 2, 0).detach().numpy())
        plt.axis('off')
    plt.subplot(1, 11, 11)
    plt.imshow(im2.cpu().permute(1, 2, 0).detach().numpy())
    plt.axis('off')
    plt.savefig('interpolations.png')
    plt.close()

if __name__ == "__main__":
    main()
