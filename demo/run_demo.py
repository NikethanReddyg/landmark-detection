import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from backend.app.model.custom_model import SimpleLandmarkDetector

# Config
IMAGE_DIR = "dataset"
ANNOTATION_SAMPLE = [
    "Eiffel_Tower/001.jpg",
    "Taj_Mahal/002.jpg"
]
CLASS_NAMES = ["Eiffel_Tower", "Taj_Mahal"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleLandmarkDetector(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("backend/app/model/custom_model_weights.pth", map_location=DEVICE))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def run_inference(img_path):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        class_logits, bbox_preds = model(img_tensor)
        class_idx = torch.argmax(class_logits, dim=1).item()
        bbox = bbox_preds.squeeze().cpu().numpy() * 224  # Denormalize

    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1 - 10), CLASS_NAMES[class_idx], fill="red")

    image.show()

if __name__ == "__main__":
    for filename in ANNOTATION_SAMPLE:
        img_path = os.path.join(IMAGE_DIR, filename)
        print(f"🔍 Running inference on: {filename}")
        run_inference(img_path)
