import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as img
from PIL import Image
import numpy as np 
from torchvision.transforms import ToTensor, ToPILImage
import glob

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

model_path = 'your model path'  #<--- best

model=MY_FINAL_MODEL()
model.to(device)

test_file_list=[]
test_folder_path='test image folder path'
submit_folder_path='submit image folder path'

test_file_list=glob.glob(test_folder_path+"/*.png")

PIL_TO_TENSER = ToTensor()

for img_path in test_file_list:
  with torch.no_grad():
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    PIL_img = Image.open(img_path).convert('RGB') # 3 720 120
    test_tensor = PIL_TO_TENSER(PIL_img)
    test_tensor = test_tensor.unsqueeze(0).to(device)
    preds = model(test_tensor)

    preds_img = preds.squeeze()
    preds_img = preds_img.permute(1,2,0).cpu().numpy()
    preds_img=NormalizeData(preds_img)
    test_img_name=str(img_path[46:])
    test_img_name=test_img_name.replace(".png","")

    output_img_name=test_img_name+"_result.png"
    matplotlib.image.imsave(submit_folder_path+output_img_name, preds_img)
    print(output_img_name + " saved!")

print("done")
