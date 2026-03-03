import torch

def extract():
    input_path = "sac_transformer_best.pth"
    output_path = "transformer_only.pth"
    
    print(f"Reading {input_path}...")
    try:
        checkpoint = torch.load(input_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
        
    if "encoder" in checkpoint:
        encoder_state = checkpoint["encoder"]
        # Strip _orig_mod. prefix just in case the BC model was compiled when it was saved
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in encoder_state.items()}
        
        torch.save({"encoder": clean_state}, output_path)
        print(f"Successfully extracted {len(clean_state)} tensors from encoder.")
        print(f"Saved the clean Transformer to: {output_path}")
    else:
        print("Error: 'encoder' key not found in the checkpoint.")

if __name__ == "__main__":
    extract()
