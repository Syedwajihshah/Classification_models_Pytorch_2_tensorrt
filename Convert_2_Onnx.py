import cv2
import onnx
import torch
from albumentations import (Compose,Resize,)
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensor
from torchvision import models
import torch.nn as nn



# class_names = image_datasets['train'].classes
class_names=open('./labels.txt')
class_names=class_names.read()
class_names=class_names.split('\n')
class_names.remove('')
print('Number of classes ',len(class_names))




def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        Resize(64, 64, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ])

    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(image=input_img)["image"]
    # prepare batch
    batch_data = torch.unsqueeze(input_data, 0)

    return batch_data



def main():
    # load pre-trained model -------------------------------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet101(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    model.load_state_dict(torch.load('./weights/resnet101_f.pth'))
    print('model')
    # preprocessing stage ----------------------------------------------------------------------------------------------
    input = preprocess_image("turkish_coffee.jpg").cuda()

    print(input.shape)


    # convert to ONNX --------------------------------------------------------------------------------------------------
    ONNX_FILE_PATH = "./weights/resnet101_f.onnx"

    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"],
                      verbose=False,output_names=["output"], export_params=True)

    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)


if __name__ == '__main__':
    main()
