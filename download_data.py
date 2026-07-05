import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("livestock").project("livestock_detection")
version = project.version(4)
dataset = version.download("yolov8", location="data/livestock")

print("Dataset downloaded to data/livestock/")
