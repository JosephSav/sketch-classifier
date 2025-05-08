import gradio as gr
from PIL import Image, ImageOps
import torch
import numpy as np
import torchvision.transforms as transforms
from modules import CNN
from dataset import SketchDataset  # For class names
from torch import nn

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names included
class_names = ["cat", "apple", "key", "bed", "basketball", "cake", "cloud", "crown", "duck", "fish"]

# Load the model
model = CNN(num_classes=len(class_names)).to(device)
state_dict = torch.load("best_model.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Preprocessing that mimics QuickDraw
def preprocess_like_quickdraw(image_pil, target_size=28, canvas_size=256, padding=16):
    """
    Simulate QuickDraw's vector-to-raster processing pipeline.
    """
    # Convert to RGBA
    image_pil = image_pil.convert("RGBA")

    # Ensure the background is white (fill the entire image with white)
    image_pil = Image.alpha_composite(Image.new("RGBA", image_pil.size, (255, 255, 255, 255)), image_pil)

    # Convert from RGBA to RGB (remove alpha channel)
    image_pil = image_pil.convert("RGB")  # Now it will be 3 channels

    # Convert to grayscale (black strokes on white background)
    image_pil = image_pil.convert("L")  # Convert to grayscale (L mode)

    # Resize to simulate large canvas (e.g., 256x256)
    image_pil = image_pil.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)

    # Invert colors (white strokes on black background)
    image_pil = ImageOps.invert(image_pil)

    # Add padding
    padded_size = canvas_size + 2 * padding
    new_img = Image.new("L", (padded_size, padded_size), 0)
    new_img.paste(image_pil, (padding, padding))

    # Center drawing by cropping to bounding box
    bbox = new_img.getbbox()
    if bbox:
        new_img = new_img.crop(bbox)
    else:
        return Image.new("L", (target_size, target_size), 0)

    # Resize to 28x28
    final_img = new_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return final_img

# Prediction function
def predict_sketch(image):
    try:
        # Extract composite image from Sketchpad (ensure we get correct data format)
        image_data = image.get('composite', None)
        if image_data is None:
            raise ValueError("No 'composite' image found in input.")

        # Debug: Check the shape of the incoming data
        print(f"Image data shape: {image_data.shape}")  # Ensure the image data is being received properly
        
        # Convert numpy array to PIL image
        image_pil = Image.fromarray(image_data.astype(np.uint8))

        # Preprocess to match training conditions
        image_pil = preprocess_like_quickdraw(image_pil)

        # Convert to tensor
        image_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)
        print(f"Image data shape post-process:  {image_tensor.shape}")

        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).item()

        # Print the prediction and logits to the console
        logits = output.squeeze(0).cpu()  # Logits for the image
        sftmx = nn.Softmax()
        probs = sftmx(logits).numpy() # Extract probabilistic predictions
        print("Class Probabilities:")
        for class_name, prob in zip(class_names, probs):
            print(f"{class_name}: {prob:.4f}")

        print(f"Predicted Class: {class_names[pred]} (Prob: {probs[pred]:.4f})")

        # Return the prediction. image_pil can be returned as output for debugging
        return f"Predicted: {class_names[pred]} ({probs[pred]:.4f}%)"

    except Exception as e:
        print("Prediction error:", e)
        return f"Error: {str(e)}", None



def clear_sketch():
    return np.zeros((560, 560, 4), dtype=np.uint8) * 255

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            sketchpad = gr.Sketchpad(canvas_size=(560, 560), label="Draw here", height=560, width=560)
        with gr.Column(scale=1):
            prediction_output = gr.Label(label="Prediction")
            predict_button = gr.Button("Predict")
            clear_button = gr.Button("Clear")

    # Bind buttons
    predict_button.click(fn=predict_sketch, inputs=sketchpad, outputs=prediction_output)
    clear_button.click(fn=clear_sketch, inputs=[], outputs=[sketchpad])


# Launch
if __name__ == "__main__":
    print("Launching Sketch Classifier UI...")
    demo.launch()



