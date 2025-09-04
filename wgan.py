import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# ========= Hyperparameters =========
latent_dim = 100
img_size = 64      # সব ছবি 64x64 এ resize
channels = 3       # RGB
batch_size = 8     # ছোট dataset -> ছোট batch size
lr = 0.0002
epochs = 300
lambda_gp = 10
n_critic = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Dataset with Augmentation =========
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(root="GAN_image", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========= Generator =========
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),   # 1x1 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),          # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),           # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),            # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),      # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# ========= Critic =========
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

# ========= Gradient Penalty =========
def gradient_penalty(critic, real_imgs, fake_imgs):
    batch_size = real_imgs.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (epsilon * real_imgs + (1 - epsilon) * fake_imgs).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ========= Initialize =========
generator = Generator().to(device)
critic = Critic().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

os.makedirs("generated_images1", exist_ok=True)

# ========= Training =========
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(device)

        # ---- Train Critic ----
        for _ in range(n_critic):
            z = torch.randn(real_imgs.size(0), latent_dim, 1, 1, device=device)
            fake_imgs = generator(z).detach()

            real_validity = critic(real_imgs)
            fake_validity = critic(fake_imgs)
            gp = gradient_penalty(critic, real_imgs, fake_imgs)

            loss_C = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            optimizer_C.zero_grad()
            loss_C.backward()
            optimizer_C.step()

        # ---- Train Generator ----
        z = torch.randn(real_imgs.size(0), latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        loss_G = -torch.mean(critic(gen_imgs))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"[Epoch {epoch}/{epochs}] [D loss: {loss_C.item():.4f}] [G loss: {loss_G.item():.4f}]")
    save_image(gen_imgs[:25], f"generated_images1/{epoch}.png", nrow=5, normalize=True)
