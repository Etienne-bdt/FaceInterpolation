import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * 8 * 4 * 4, latent_dim * 2)  # Output mean and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 8 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim * 8, 4, 4)),
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Assuming input is normalized to [0, 1]
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encoder
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        # Decoder
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    def interpolate(self, x1, x2, num_steps=10):
        with torch.no_grad():
            mu1, logvar1 = self.encoder(x1).chunk(2, dim=1)
            mu2, logvar2 = self.encoder(x2).chunk(2, dim=1)
            z1 = self.reparameterize(mu1, logvar1)
            z2 = self.reparameterize(mu2, logvar2)
            interpolated_images = []
            for alpha in torch.linspace(0, 1, num_steps):
                z_interp = (1 - alpha) * z1 + alpha * z2
                x_interp = self.decoder(z_interp)
                interpolated_images.append(x_interp)
        return torch.stack(interpolated_images)
    
if __name__ == "__main__":
    # Example usage
    input_dim = 1  # Grayscale images
    hidden_dim = 32
    latent_dim = 128
    vae = VAE(input_dim, hidden_dim, latent_dim)
    
    # Dummy input
    x = torch.randn(16, input_dim, 64, 64)  # Batch of 16 images
    x_recon, mu, logvar = vae(x)
    
    print("Reconstructed shape:", x_recon.shape)
    print("Mean shape:", mu.shape)
    print("Log variance shape:", logvar.shape)