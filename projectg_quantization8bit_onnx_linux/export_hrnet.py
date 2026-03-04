import torch
import torch.onnx
import sys
import os

sys.path.insert(0, os.path.join("reference", "project7"))
from model import get_hrnet_w18
sys.path.pop(0)

def export_hrnet():
    checkpoint_path = os.path.join("reference", "project7", "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    print("Loading HRNet model...")
    model = get_hrnet_w18(num_classes=10, in_channels=1)
    
    device = torch.device('cpu') # Export on CPU
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Input shape is (B, C, H, W) -> (1, 1, 480, 640)
    dummy_input = torch.randn(1, 1, 480, 640, device=device)
    
    onnx_path = "hrnet.onnx"
    print(f"Exporting HRNet to {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        fallback=True,
        input_names=['image_input'],
        output_names=['vision_features']
    )
    
    print("Done! Exported hrnet.onnx")

if __name__ == "__main__":
    export_hrnet()
