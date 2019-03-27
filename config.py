from easydict import EasyDict as edict
# from pathlib import Path
import torch
import os
from torchvision import transforms as trans
from utils.constants import *
list_model = ['wget https://www.dropbox.com/s/akktsgxp0n8cwn2/model_mobilefacenet.pth?dl=0 -O model_mobilefacenet.pth',
'wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O model_ir_se50.pth',
'wget https://www.dropbox.com/s/rxavczg9dlxy3a8/model_ir50.pth?dl=0 -O model_ir50.pth']
def get_config(mode = 'app', net_size = 'large', net_mode = 'ir_se', use_mtcnn = 1, threshold = 1.25):
    conf = edict()
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.input_size = [112, 112]
    conf.face_limit = 5 
    conf.min_face_size = 30 
    if mode =='app':
        assert net_size in ['mobi', 'large'], 'net_size should be mobi or large, please change in cogfig.py'
        conf.use_tensor = True
        conf.work_path = WORK_PATH
        conf.model_path = '%s/models'%WORK_PATH
        conf.log_path = '%s/log'%WORK_PATH
        conf.save_path = '%s/save'%WORK_PATH
        conf.facebank_path = '%s/Face_bank'%WORK_PATH
        
        conf.threshold = threshold
        if use_mtcnn:
            conf.use_mtcnn = True
        else:
            conf.use_mtcnn = False
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.test_transform = trans.Compose([
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
        if net_size == 'large':
            conf.use_mobilfacenet = False
            if net_mode == 'ir_se':
                conf.net_mode = 'ir_se' # or 'ir'
                conf.weight_path = '%s/weights/model_ir_se50.pth'%WORK_PATH
                conf.url = list_model[1]
            else:
                conf.net_mode = 'ir' # or 'ir'
                conf.weight_path = '%s/weights/model_ir50.pth'%WORK_PATH
                conf.url = list_model[2]
        if net_size =='mobi':
            conf.use_mobilfacenet = True
            conf.weight_path = '%s/weights/model_mobilefacenet.pth'%WORK_PATH
            conf.url = list_model[0]

    if mode =='training_eval':
        conf.lr = 1e-3
        conf.milestones = [18,30,42]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.train_root = "/mnt/01D4A1D481139570/Dataset/Face/casia"
        conf.file_list =  '/mnt/01D4A1D481139570/Dataset/Face/casia_train.txt' 
        conf.batch_size = 4
        conf.lfw_root = '/mnt/01D4A1D481139570/Dataset/Face/data/LFW/lfw_align_112'
        conf.lfw_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/LFW/pairs.txt'
        conf.agedb_root = '/mnt/01D4A1D481139570/Dataset/Face/data/AgeDB-30/agedb30_align_112'
        conf.agedb_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/AgeDB-30/agedb_30_pair.txt'
        conf.cfp_root = '/mnt/01D4A1D481139570/Dataset/Face/data/CFP-FP/CFP_FP_aligned_112'
        conf.cfp_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/CFP-FP/cfp_fp_pair.txt'
    return conf