import torch
import torch.nn as nn
import torch.onnx
import os

from sac_trainer import SACTrainer, SACConfig
from transformer_encoder import SpatioTemporalEncoder

class ActorInferenceWrapper(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        
    def forward(self, features):
        mean, std, logits = self.actor(features)
        
        # Using the same logic as SACTrainer.select_action(evaluate=True)
        guess = torch.sigmoid(mean)
        trigger_idx = torch.argmax(logits, dim=1)
        trigger_val = trigger_idx.float().unsqueeze(1)
        
        action = torch.cat([guess, trigger_val], dim=1)
        return action

def export_models():
    checkpoint_path = "sac_coc_best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    # To initialize the correct shapes, we can mock the Config and initialize Trainer
    cfg = SACConfig(history_len=10)
    
    # Obs_dim and Action_dim according to train.py
    obs_dim = 1
    action_dim = 2
    input_dim = obs_dim * cfg.history_len + action_dim * cfg.history_len # 30
    
    # Initialize the entire Trainer to get fully initialized components
    print("Initializing SACTrainer to load weights...")
    trainer = SACTrainer(cfg, obs_dim=input_dim, action_dim=action_dim)
    trainer.load(checkpoint_path)
    print("Weights loaded successfully.")
    
    # 1. Export Transformer
    print("Exporting Transformer...")
    # transformer expects obs_stack of dim 33020
    transformer = trainer.encoder
    transformer.eval()
    dummy_obs_stack = torch.randn(1, 33020, device=trainer.device)
    
    torch.onnx.export(
        transformer,
        dummy_obs_stack,
        "transformer.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        fallback=True,
        input_names=['obs_stack'],
        output_names=['encoded_features']
    )
    print("Done! Exported transformer.onnx")

    # 2. Export Actor wrapper
    print("Exporting SAC Actor...")
    actor = trainer.actor
    actor.eval()
    
    actor_wrapper = ActorInferenceWrapper(actor)
    actor_wrapper.eval()
    
    # The Transformer outputs 52 dimensional feature vector
    dummy_actor_input = torch.randn(1, 52, device=trainer.device)
    
    torch.onnx.export(
        actor_wrapper,
        dummy_actor_input,
        "sac_actor.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        fallback=True,
        input_names=['features'],
        output_names=['action'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    print("Done! Exported sac_actor.onnx")

if __name__ == "__main__":
    export_models()
