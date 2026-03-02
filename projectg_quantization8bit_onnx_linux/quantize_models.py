import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_all():
    models = ['hrnet.onnx', 'transformer.onnx', 'sac_actor.onnx']
    
    for model_name in models:
        quantized_name = model_name.replace(".onnx", "_int8.onnx")
        if not os.path.exists(model_name):
            print(f"Warning: {model_name} does not exist. Please run the export scripts first.")
            continue
            
        print(f"Applying Dynamic Quantization to {model_name}...")
        try:
            quantize_dynamic(
                model_input=model_name,
                model_output=quantized_name,
                weight_type=QuantType.QUInt8,
            )
            print(f"Saved {quantized_name}")
        except Exception as e:
            print(f"Failed to quantize {model_name}: {e}")

if __name__ == "__main__":
    quantize_all()
