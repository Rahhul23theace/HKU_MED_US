



import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import random

def load_model(model_path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 5),
        nn.Sigmoid()
    )

    device = torch.device("cpu")

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def single_model_prediction(model, inputs):
    outputs = model(inputs)
    return outputs

def load_image_from_file(file, transform=None):
    image = Image.open(file).convert('RGB')
    if transform:
        image = transform(image)
    return image




def generate_feedback(predicted_classes):
    feedback = []
    for prediction in predicted_classes:
        feedback_for_prediction = []
        labels = prediction.split(", ")
        for label in labels:
            if label == "Artefact":
                feedback_for_prediction.append('''You can avoid acoustic shadowing from the ribs by asking the subject to take a 
                                               deep inspiration, which will lower the diaphragm and hence the position of the
                                                kidney away from the ribcage, or by positioning the probe in between the ribs to 
                                               avoid the artefact. Fasting or gently applying pressure on the probe to displace 
                                               the gas away from the area may partially overcome ring-down artefacts from bowel gas.
                                                <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>''')
                
            elif label == "Incorrect_Gain":
                feedback_for_prediction.append('''The image is either too "bright" or too "dark". <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a> or
                                                <a href="https://123sonography.com/blog/ultrasound-101-part-5-gain-and-time-gain-compensation#:~:text=What%20is%20gain%3F,much%20each%20echo%20is%20amplified." target="_blank">Visit Sonography123</a>''')
                
            elif label == "Incorrect_Position":
                feedback_for_prediction.append('''The kidney is not centrally placed or incompletely imaged. 
                                               Having the subject in decubitus position helps to get good access to 
                                               image the kidney (<a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>) ''')
                
            elif label == "Optimal":
                feedback_for_prediction.append('''Well done/good work for obtaining optimal image quality of the kidney.''')
                
            elif label == "Wrong":
                feedback_for_prediction.append('''The image acquired is not of the kidney. <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>''')
            else:
                feedback_for_prediction.append('''I am sorry, something went wrong''')
        feedback.append(feedback_for_prediction)
    return feedback
