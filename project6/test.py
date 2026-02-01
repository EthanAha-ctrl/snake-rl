import cv2
import time
from coc_env import CoCEnv

def main():
    # Initialize the environment with human render mode
    env = CoCEnv(render_mode='human')
    
    # Reset the environment
    obs, info = env.reset()
    print("Environment reset. Initial observation:", obs)
    
    terminated = False
    truncated = False
    step_count = 0
    
    # Simple loop for an episode
    while not (terminated or truncated):
        step_count += 1
        
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_count}: Action={action}, Obs={obs}, Reward={reward}, Terminated={terminated}")
        
        # Render
        env.render()
        
        # Pause every frame as requested
        print("Paused. Press any key on the image window to continue...")
        key = cv2.waitKey(0) # Wait indefinitely for a key press
        if key == 27: # ESC to quit
            break
            
    print("Episode finished.")
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
