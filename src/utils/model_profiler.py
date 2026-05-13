import thop
import time
import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info

def measure_model_metrics(model, input_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    
    print("="*50)
    print("Model Parameter Statistics")
    print("="*50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n" + "="*50)
    print("Calculate FLOPs using thop")
    print("="*50)
    
    dummy_input = torch.randn(input_size).to(device)
    flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
    
    print(f"FLOPs: {flops:,} ({flops/1e9:.2f}G)")
    print(f"Parameters (thop): {params:,} ({params/1e6:.2f}M)")
    
    try:
        print("\n" + "="*50)
        print("Calculate FLOPs using ptflops")
        print("="*50)
        
        macs, params_pt = get_model_complexity_info(
            model, 
            tuple(input_size[1:]),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
        print(f"MACs: {macs}")
        print(f"Parameters (ptflops): {params_pt}")
        
        # FLOPs ≈ 2 * MACs (for most operations)
        flops_pt = 2 * eval(macs.split()[0]) * float(macs.split()[1].replace('G', '').replace('M', ''))
        print(f"Estimated FLOPs: {flops_pt:.2f}G")
    except:
        print("ptflops calculation failed, skipping...")
    
    # 4. Inference time test
    print("\n" + "="*50)
    print("Inference Time Test")
    print("="*50)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Formal test
    if device == 'cuda':
        torch.cuda.synchronize()
    
    num_runs = 100
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start_time)
    
    # Statistical results
    times = np.array(times)
    mean_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000
    fps = 1000 / mean_time
    
    print(f"Average Inference Time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    # 5. Memory usage (optional)
    if device == 'cuda':
        print("\n" + "="*50)
        print("GPU Memory Usage")
        print("="*50)
        torch.cuda.reset_peak_memory_stats()
        _ = model(dummy_input)
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU Memory: {memory_used:.2f} MB")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'inference_time_ms': mean_time,
        'fps': fps
    }

# Example: Define a simple CNN model for testing
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Create model
    model = SimpleCNN(num_classes=10)
    input_size = (1, 3, 32, 32)  # (batch, channels, height, width)
    
    # Measure model metrics
    metrics = measure_model_metrics(model, input_size)
    
    # Print model architecture using torchsummary
    print("\n" + "="*50)
    print("Model Architecture")
    print("="*50)
    summary(model, input_size=(3, 32, 32), device='cpu')