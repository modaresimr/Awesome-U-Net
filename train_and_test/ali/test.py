import torch
import torch.nn as nn

# Define a modified dummy model for demonstration


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()  # New layer to flatten the tensor before FC
        self.fc = nn.Linear(32 * 224 * 224, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)  # Flatten the tensor
        x = self.fc(x)
        return x


# Create an instance of the modified dummy model
model = DummyModel()

# Dummy input for profiling (change it according to your model's input shape)
dummy_input = torch.randn(1, 3, 224, 224)

# Function to count FLOPs for a specific layer


def count_flops(layer, input_size, output_size):
    flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * output_size[0] * output_size[1]
    return flops


# Enable profiling
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # Temporarily change the model's input size to match dummy_input
    original_input_size = model.conv1.weight.size()[1:]
    model.conv1.weight = nn.Parameter(torch.randn(16, 3, 3, 3))
    model.fc = nn.Linear(32 * 224 * 224, 10)

    # Forward pass to profile FLOPs
    model(dummy_input)

    # Reset the model's input size back to the original
    model.conv1.weight = nn.Parameter(torch.randn(*original_input_size))
    model.fc = nn.Linear(32 * 224 * 224, 10)

# Compute the GFLOPS estimation manually
total_flops = 0
for layer in model.children():
    if isinstance(layer, nn.Conv2d):
        input_size = dummy_input.size()[2:]
        flops = count_flops(layer, input_size, (224, 224))  # Since the input size is fixed, use the original 224x224 size
        total_flops += flops

gflops = total_flops / prof.self_cpu_time_total / 1e9  # Divide by total time in seconds and convert to GFLOPS

print(f"GFLOPS: {gflops:.20f}")
