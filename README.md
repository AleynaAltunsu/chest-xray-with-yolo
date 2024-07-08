# chest-xray-with-yolo
---

# YOLOv10 Object Detection Project

This project demonstrates how to use the YOLOv10 model for object detection. The steps include downloading a dataset from Roboflow, modifying the dataset configuration, organizing directories, training the YOLO model, and performing inference on test images.

## Prerequisites

- Python 3.6+
- Roboflow Python package
- Ultralytics YOLO package
- PyYAML package
- Matplotlib package

You can install the required packages using the following commands:

```bash
pip install roboflow ultralytics pyyaml matplotlib
```

## Project Structure

```
YOLOv10-Object-Detection/
├── data.yaml                 # Dataset configuration file
├── yolov10l.yaml             # YOLOv10 model configuration file
└── script.py                 # Main script for training and inference
```

## Steps

### 1. Download the Dataset

The dataset is downloaded from Roboflow using the provided API key. The dataset is downloaded as a ZIP file and extracted.

### 2. Modify Dataset Configuration

The dataset configuration (`data.yaml`) is modified to point to the correct paths for training and validation images.

### 3. Organize Directories

The script organizes the directories to ensure the dataset is in the correct location for training.

### 4. Train the YOLO Model

The YOLOv10 model is trained using the specified dataset configuration and training parameters.

### 5. Perform Inference

The trained model is used to perform inference on a test image. The results are plotted and displayed using Matplotlib.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/AleynaAltunsu/YOLOv10-Object-Detection.git
cd YOLOv10-Object-Detection
```

2. Modify the `script.py` file to include your Roboflow API key:

```python
rf = Roboflow(api_key="YOUR_API_KEY")
```

3. Run the script:

```bash
python script.py
```

## script.py

Here is the main script for training and inference:

```python
from ultralytics import YOLO
import os
import yaml
import shutil
from roboflow import Roboflow
import matplotlib.pyplot as plt

# Disable WandB logging
os.environ['WANDB_DISABLED'] = 'true'

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="YOUR_API_KEY")
dataset = rf.workspace("pr-ea2pm").project("pr_proj").version(2).download("yolov9")

# Load and modify YAML file
yaml_path = '/kaggle/working/Pr_Proj-2/data.yaml'
with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)
data['train'] = '../train/images'
data['val'] = '../valid/images'
with open(yaml_path, 'w') as file:
    yaml.dump(data, file)

# Print the modified YAML data
with open(yaml_path, 'r') as file:
    print(yaml.safe_load(file))

# Organize directories
working_dir = '/kaggle/working/'
existing_folder = os.path.join(working_dir, 'Pr_Proj-2')
new_folder = os.path.join(working_dir, 'datasets/Pr_Proj-2')

# Remove the destination folder if it exists
if os.path.exists(new_folder):
    shutil.rmtree(new_folder)

# Move the existing folder to the new location
shutil.move(existing_folder, os.path.join(working_dir, 'datasets'))

# Train YOLO model
model = YOLO('yolov10l.yaml')
model.train(data='/kaggle/working/datasets/Pr_Proj-2/data.yaml', epochs=35, imgsz=640)

# Load the best model and perform inference
best_model = YOLO('/kaggle/working/runs/detect/train/weights/last.pt')
results = best_model('/kaggle/working/datasets/Pr_Proj-2/test/images/00030279_000_png.rf.6594da797e69f6c360c2aff4ee883fcf.jpg')

# Plot and display results
plt.imshow(results[0].plot())
plt.show()
```

## Repository

The project repository is available at [https://github.com/AleynaAltunsu/YOLOv10-Object-Detection](https://github.com/AleynaAltunsu/YOLOv10-Object-Detection).

---

Replace `"YOUR_API_KEY"` with your actual Roboflow API key in the `script.py` file.
