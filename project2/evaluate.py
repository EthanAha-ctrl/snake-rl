import torch
import gymnasium as gym
import numpy as np
import torchvision.transforms as transforms
from train import get_model, CartPoleDataset
import cv2
import json
import os
import pickle


def evaluate_model_live(model_path="position_detection.pth"):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.unwrapped.theta_threshold_radians = np.deg2rad(45.0)
    env.unwrapped.x_threshold = 1.5

    state, _ = env.reset()

    initial_frame = env.render()
    img_h, img_w, _ = initial_frame.shape

    try:
        with open("data/dataset_config.json", "r") as f:
            config = json.load(f)
            x_threshold = config["x_threshold"]
    except FileNotFoundError:
        print("Warning: data/dataset_config.json not found. Using default x_threshold=2.4")
        x_threshold = 2.4

    crop_size = min(img_h, img_w)

    model = get_model(input_size=crop_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(crop_size),
    ])

    pos_errors = []
    angle_errors = []
    steps_collected = 0
    target_steps = 1000

    done = False
    while steps_collected < target_steps:
        if done:
            state, _ = env.reset()
            done = False

        image = env.render()

        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()[0]

        actual_cart_position = state[0]
        actual_pole_angle = state[2]

        world_width = x_threshold * 2
        scale = img_w / world_width
        predicted_x = int(prediction[0] * scale + img_w / 2.0)

        vis_image = image.copy()
        cv2.line(vis_image, (predicted_x, 0), (predicted_x, 400), (255, 0, 0), 2)

        # Draw predicted angle
        predicted_angle = prediction[1]
        pole_length = 100
        cart_y = 300  # valid for standard 400px height
        end_x = int(predicted_x + pole_length * np.sin(predicted_angle))
        end_y = int(cart_y - pole_length * np.cos(predicted_angle))
        cv2.line(vis_image, (predicted_x, cart_y), (end_x, end_y), (255, 0, 0), 2)

        cv2.imshow("CartPole-v1", vis_image)
        cv2.waitKey(30)

        print(f"Predicted: Cart Position={prediction[0]:.2f}, Pole Angle={prediction[1]:.2f}")
        print(f"Actual:    Cart Position={actual_cart_position:.2f}, Pole Angle={actual_pole_angle:.2f}")
        print("-" * 30)

        pos_errors.append(prediction[0] - actual_cart_position)
        angle_errors.append(prediction[1] - actual_pole_angle)
        steps_collected += 1

        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    cv2.destroyAllWindows()

    if pos_errors and angle_errors:
        error_data = {
            "pos_errors": pos_errors,
            "angle_errors_deg": list(np.degrees(angle_errors)),
        }
        with open("error_metrics.pkl", "wb") as f:
            pickle.dump(error_data, f)
        print(f"Saved error distributions to error_metrics.pkl (steps collected: {steps_collected})")



if __name__ == '__main__':
    try:
        evaluate_model_live(model_path="position_detection.pth")
    except FileNotFoundError:
        print(f"Error: position_detection.pth not found. Please train the model first.")
    except Exception as e:
        print(f"An error occurred: {e}")
