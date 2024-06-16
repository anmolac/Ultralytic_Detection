from ultralytics import YOLO
from torchvision.transforms import ColorJitter
from PIL import Image

# Load the model
model = YOLO('yolov8n.pt')

# Run predictions
results = model('phototest.png')

# Plot results with a custom bounding box line thickness
results[0].plot  # Adjust line_thickness as needed
results[0].show()

results.save('E:\Python\Yolo_V8\runs\save_test1.png') 
