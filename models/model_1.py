from torch import nn
# L'immagine è 224x224 e si passa da 3 canali rgb

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.linear_relu_stack = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), # 112
          nn.BatchNorm2d(num_features=16),
          nn.LeakyReLU(),
          nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), # 56
          nn.BatchNorm2d(num_features=32),
          nn.LeakyReLU(),
          nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), # 28
          nn.BatchNorm2d(num_features=64),
          nn.LeakyReLU(),
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.BatchNorm2d(num_features=128),
          nn.LeakyReLU(),
          nn.Conv2d(128, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(num_features=256),
          nn.LeakyReLU(),
          nn.AvgPool2d(4, 4), # Add a max pooling layer
          nn.Flatten(),
          nn.Linear(256*7**2, 200),  # 200 output classes
        )

    def forward(self, x):
        # Define forward pass
        return self.linear_relu_stack(x)