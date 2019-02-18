from easydict import EasyDict as edict
from pathlib import Path
import torch

def get_config():
    conf = edict()
    conf.data_path = Path('data')
    conf.input_size = [112,112]
    conf.use_mobilfacenet = False
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.facebank_path = conf.data_path/'facebank'
    conf.threshold = 1.34
    conf.face_limit = 10 
    conf.use_mtcnn = False
    #when inference, at maximum detect 10 faces in one image, my laptop is slow
    conf.min_face_size = 30 
    conf.facebank_path = conf.data_path/'facebank'
    # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf
def get_config_train(training = True):
    conf = edict()
    conf.data_path = Path('/mnt/01D4A1D481139570/Dataset/Face/faces_ms1m-refine-v2_112x112')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.batch_size = 32 # irse net depth 50    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [12,15,18]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.train_root = "/mnt/01D4A1D481139570/Dataset/Face/faces_ms1m-refine-v2_112x112"
        conf.file_list =  '/mnt/01D4A1D481139570/Dataset/Face/faces_ms1m-refine-v2_112x112/ms1m_list_112x112.txt'   
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
        conf.lfw_root = '/mnt/01D4A1D481139570/Dataset/Face/data/lfw_process/lfw-112X96'
        conf.lfw_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/lfw_process/pairs.txt'
        conf.agedb_root = '/mnt/01D4A1D481139570/Dataset/Face/data/AgeDB-30/agedb30_align_112'
        conf.agedb_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/AgeDB-30/agedb_30_pair.txt'
        conf.cfp_root = '/mnt/01D4A1D481139570/Dataset/Face/data/CFP-FP/CFP_FP_aligned_112'
        conf.cfp_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/CFP-FP/cfp_fp_pair.txt'
    return conf