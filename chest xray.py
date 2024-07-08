from ultralytics import YOLO

import os 
from urllib.request import urlretrieve
from zipfile import ZipFile
import yaml
import shutil
from roboflow import Roboflow
import matplotlib.pyplot as plt

import os

os.environ['WANDB_DISABLED'] = 'true'

rf = Roboflow(api_key="FiJucU9399Jud0RZRLWj")
project = rf.workspace("pr-ea2pm").project("pr_proj")
version = project.version(2)
dataset = version.download("yolov9")

# Load the YAML file
with open('/kaggle/working/Pr_Proj-2/data.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Modify the value of datasets_dir
data['train'] = '../train/images'
data['val'] = '../valid/images'

# Save the modified data back to the YAML file
with open('/kaggle/working/Pr_Proj-2/data.yaml', 'w') as file:
    yaml.dump(data, file)


with open('/kaggle/working/Pr_Proj-2/data.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Now you can work with the data loaded from the YAML file
print(data)


# Specify the paths
working_dir = '/kaggle/working/'
existing_folder_name = 'Pr_Proj-2'
new_folder_name = 'datasets'

# Create the new folder
new_folder_path = os.path.join(working_dir, new_folder_name)
os.makedirs(new_folder_path, exist_ok=True)

# Move the existing folder into the new folder
existing_folder_path = os.path.join(working_dir, existing_folder_name)
shutil.move(existing_folder_path, new_folder_path)


model = YOLO('yolov10l.yaml')

results = model.train(data='/kaggle/working/datasets/Pr_Proj-2/data.yaml', epochs=35, imgsz=640)

best_model = YOLO('/kaggle/working/runs/detect/train/weights/last.pt')

results = model(['/kaggle/working/datasets/Pr_Proj-2/test/images/00030279_000_png.rf.6594da797e69f6c360c2aff4ee883fcf.jpg'])  # return a list of Results objects
result = results[0]

plt.imshow(result.plot())