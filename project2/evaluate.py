import torch
import gymnasium as gym
import numpy as np
import torchvision.transforms as transforms
from train import get_model, CartPoleDataset
import cv2
import argparse
from torch.utils.data import DataLoader

def evaluate_model_live(model_path="position_detection.pth"):
    """
    Evaluates the trained perception model and visualizes the results in a live environment.

    Args:
        model_path (str): Path to the trained model.
    """
    # Load the model
    model = get_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the transformations for the input images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset()
    
    done = False
    while not done:
        # Render the environment
        image = env.render()
        
        # Preprocess the image and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()[0]

        # Get the actual values
        actual_cart_position = state[0]
        actual_pole_angle = state[2]
        
        # Draw the predicted position on the image
        scale = 600 / 4.8
        predicted_x = int(prediction[0] * scale + 600 / 2.0)
        
        # Draw a red vertical line for the predicted cart position
        vis_image = image.copy()
        cv2.line(vis_image, (predicted_x, 0), (predicted_x, 400), (255, 0, 0), 2)

        # Display the image with the prediction
        cv2.imshow("CartPole-v1", vis_image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Print the results
        print(f"Predicted: Cart Position={prediction[0]:.2f}, Pole Angle={prediction[1]:.2f}")
        print(f"Actual:    Cart Position={actual_cart_position:.2f}, Pole Angle={actual_pole_angle:.2f}")
        print("-" * 30)

        # Take a random action to proceed
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    cv2.destroyAllWindows()

def evaluate_model_on_dataset(model_path="position_detection.pth", data_dir="data", batch_size=32):
    """
    Evaluates the trained perception model on a dataset.

    Args:
        model_path (str): Path to the trained model.
        data_dir (str): Directory with the test data.
        batch_size (int): Batch size for evaluation.
    """
    # Load the model
    model = get_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the transformations for the input images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the dataset and dataloader
    # Shuffle is set to False for evaluation
    dataset = CartPoleDataset(data_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

    mean_loss = total_loss / num_samples if num_samples > 0 else 0
    print(f"Evaluation on dataset complete.")
    print(f"Mean Squared Error: {mean_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the trained perception model.")
    parser.add_argument("--mode", type=str, default="live", choices=["live", "dataset"],
                        help="Evaluation mode: 'live' for live environment or 'dataset' for evaluation on the dataset.")
    parser.add_argument("--model_path", type=str, default="position_detection.pth",
                        help="Path to the trained model.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory with the test data (for dataset mode).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for dataset evaluation.")
    args = parser.parse_args()

    try:
        if args.mode == "live":
            evaluate_model_live(model_path=args.model_path)
        elif args.mode == "dataset":
            evaluate_model_on_dataset(model_path=args.model_path, data_dir=args.data_dir, batch_size=args.batch_size)
    except FileNotFoundError:
        print(f"Error: {args.model_path} not found. Please train the model first.")
    except Exception as e:
        print(f"An error occurred: {e}")
