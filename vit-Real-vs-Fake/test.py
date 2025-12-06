import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification

# Inizialize the model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Load the checkpoint
checkpoint = torch.load("results/best_model.pth", map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"]

# Remove the "vit." prefix
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("vit.vit."):
        new_k = k.replace("vit.vit.", "vit.")
    else:
        new_k = k
    new_state_dict[new_k] = v

# Load the adjusted weights

model.eval()

# Prepare an image
img = Image.open("test_image.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
])
x = transform(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(x)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

print(f"Prediction: {'REAL' if pred_class == 0 else 'FAKE'} ({probs[0][pred_class]*100:.2f}%)")
