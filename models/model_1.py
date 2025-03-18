from torch import nn
# L'immagine è 224x224 e si passa da 3 canali rgb

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network

        # Add more layers...
        self.linear_relu_stack = nn.Sequential( #224
          nn.Conv2d(3, 64, kernel_size=3, padding=1, stride = 2),  #112 dimensioni della matrice singola
          nn.BatchNorm2d(num_features=64),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, padding=1, stride = 2), #56
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), #28
          nn.BatchNorm2d(num_features=256),
          nn.ReLU(),
          nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2), #14
          nn.BatchNorm2d(num_features=512),
          nn.ReLU(),
          nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2), #7
          nn.BatchNorm2d(num_features=1024),
          nn.ReLU(),
          nn.Flatten(), # linearizza 1 x 7
          # 1024: output del conv2d, 7: le dimensioni dell'immagine, 2: perché 2d
          nn.Linear(1024*7**2, 200),  # 200 output classes
        )

    def forward(self, x):
        # Define forward pass
        return self.linear_relu_stack(x)