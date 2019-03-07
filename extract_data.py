import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
from backbone.model import SE_IR, MobileFaceNet, l2_norm
import torch
import PIL.Image as Image
from config import get_config_train
from tqdm import tqdm
conf = get_config_train()
def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            
            return img
    except IOError:
        print('Cannot load image ' + path)
from torchvision import transforms as trans
class EX_MS():
    def __init__(self, root= conf['train_root'], file_list = conf.file_list, transform=None, loader=img_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root = root
        self.transform = transform
        self.loader = loader
        image_list = []
        label_list = []
        with open(file_list) as f:
            self.img_label_list = f.read().splitlines()
        for info in self.img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))
        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        self.num_iter = len(self.image_list)// 64
        print("dataset size: ", len(self.image_list), '/', self.class_nums)
        self.test_transform = trans.Compose([
                    transforms.Resize((112, 112)),
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    def extract_data(self, model):
        np_data = []
        for info in tqdm(self.img_label_list):
            image_path, label_name = info.split(' ')
            #img = self.loader(os.path.join(self.root, image_path))
            img = Image.open(os.path.join(self.root, image_path))

            data = model(self.test_transform(img).to(self.device).unsqueeze(0))

            np_data.append(data.detach().cpu().numpy())
        np.save('MS1M',np.array(np_data))
        print(np_data)

if __name__ == '__main__':
	model = SE_IR(50, 0.6, 'ir_se')
	weight = './weights/model_ir_se50.pth'
	model.load_state_dict(torch.load(weight))  
	model.cuda()
	ex = EX_MS()
	ex.extract_data(model.eval())
