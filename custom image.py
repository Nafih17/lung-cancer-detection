import torch
import torch.nn as nn
import torch.nn.functional as F
from model import UNet  # Ensure `model.py` contains the UNet implementation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CTProcessor:
    def __init__(self, model_path, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the CTProcessor with the trained model path and device.

        Args:
            model_path (str): Path to the saved model file (e.g., 'model_weights.pth').
            device (str, optional): Device to use for inference ('cuda:0' for GPU, 'cpu' for CPU).
                                    Defaults to "cuda:0" if available, otherwise "cpu".
        """
        self.device = torch.device(device)
        self.model = UNet().to(self.device)  # Move the model to the specified device
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def preprocess_image(self, image_path):
        """
        Preprocesses the input CT scan image.

        Args:
            image_path (str): Path to the CT scan image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        # Load the image
        image = Image.open(image_path).convert('L')

        # Preprocess the image (e.g., resize, normalize)
        # This part needs to be adjusted based on your specific preprocessing steps
        # used during training.
        # For example:
        # image = image.resize((256, 256))  # Resize to 256x256 pixels
        # image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return image_tensor

    def predict(self, image_path):
        """
        Predicts the lung segmentation mask for the given CT scan image.

        Args:
            image_path (str): Path to the CT scan image file.

        Returns:
            numpy.ndarray: Predicted segmentation mask.
        """
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(image_tensor)
        predicted_mask = (output > 0.5).float().cpu().squeeze(0).numpy()
        return predicted_mask

    def visualize_prediction(self, image_path):
        """
        Loads the image, performs prediction, and displays the input image and
        the predicted segmentation mask.

        Args:
            image_path (str): Path to the CT scan image file.
        """
        image = Image.open(image_path).convert('L')
        predicted_mask = self.predict(image_path)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.show()

# Example usage:
if __name__ == "__main__":
    model_path = r"C:\Users\Admin\PycharmProjects\mushfiq\32batch_best_model.pth"  # Replace with the actual path to your saved model file (e.g., "models/trained_unet_model.pth")
    processor = CTProcessor(model_path)
    custom_image_path = r"C:\Users\Admin\PycharmProjects\mushfiq\lung with no cancer.jpg"  # Replace with the actual path to your custom image
    processor.visualize_prediction(custom_image_path)