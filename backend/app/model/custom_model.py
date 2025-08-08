# backend/app/model/custom_model.py
import torch
import torch.nn as nn
import torchvision.transforms as T

# Define model architecture
class SimpleLandmarkDetector(nn.Module):
    def __init__(self, num_classes):
        super(SimpleLandmarkDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
        self.bbox_regressor = nn.Linear(64, 4)  # [x1, y1, x2, y2]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        class_logits = self.classifier(x)
        bbox = self.bbox_regressor(x)
        return class_logits, bbox

# Load model and weights
NUM_CLASSES = 10  # Update based on your dataset
model = SimpleLandmarkDetector(num_classes=NUM_CLASSES)
# model.load_state_dict(torch.load("app/model/custom_model_weights.pth", map_location="cpu"))
model.eval()

# Inference function
def run_custom_model(image_tensor):
    image_tensor = T.Resize((224, 224))(image_tensor).unsqueeze(0)  # Resize and batch
    with torch.no_grad():
        class_logits, bbox = model(image_tensor)
        scores = torch.softmax(class_logits, dim=1)
        confidence, label_idx = scores.max(dim=1)
        label = f"Class_{label_idx.item()}"  # Replace with actual class name mapping
        box = bbox.squeeze().tolist()
        return {
            "boxes": [box],
            "labels": [label],
            "scores": [confidence.item()]
        }
