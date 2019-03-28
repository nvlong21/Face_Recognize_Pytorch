from Face_Alignt.network import PNet,ONet
import torch,cv2,itertools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from Face_Alignt.matlab_cp2tform import get_similarity_transform_for_cv2
from PIL import Image
import math, os
def alignment(src_img, src_pts, default_square = True):
    ref_pts = np.array([[30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041]])
    crop_size = (112, 112)
    if crop_size[1]==112 and default_square:
        ref_pts[:,0] += 8.0

    src_pts = np.array(src_pts).reshape(5,2)
    
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img
def resize_square(img, height=128, color=(0, 0, 0)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
    dw = height - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw // 2, dh // 2

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
def get_anchors(scale=64):
    '''
    compute anchors
    return:
        u_boxes:tensor([anchor_num,4]) (cx,cy,w,h): real anchors
        boxes:tensor([anchor_num,4]) (x1,y1,x2,y2): crop box for ONet,each with size 80
    '''
    sizes = [float(s) / scale for s in [32]]
    
    aspect_ratios = [(1.,)]
    feature_map_sizes = [int(scale/16)]
    
    num_layers = len(feature_map_sizes)
    u_boxes,boxes = [],[]
    for i in range(num_layers):
        fmsize = feature_map_sizes[i]
        for h,w in itertools.product(range(fmsize),repeat=2):
            cx = float(w)/feature_map_sizes[i]
            cy = float(h)/feature_map_sizes[i]
            
            s = sizes[i]
            for j,ar in enumerate(aspect_ratios[i]):
                u_boxes.append((cx,cy,float(s)*ar,float(s)*ar))
                boxes.append((w*16-32,h*16-32,w*16+32,h*16+32))       
    return torch.Tensor(u_boxes),torch.Tensor(boxes).long()

def nms(bboxes,scores,threshold=0.35):
    '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
        else:
            i = order[0].item()
        keep.append(i) 

        if order.numel() == 1:
            break 

        xx1 = x1[order[1:]].clamp(min=x1[i]) 
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter) 
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1] 
    return torch.LongTensor(keep)
    
def decode_box(loc, size=64):
    variances = [0.1,0.2]
    anchor,crop = get_anchors(scale=size)
    cxcy = loc[:,:2] * variances[0] * anchor[:,2:] + anchor[:,:2]
    wh = torch.exp(loc[:,2:] * variances[1]) * anchor[:,2:]
    boxes = torch.cat([cxcy-wh/2,cxcy+wh/2],1)
    
    return boxes,anchor,crop
    
def decode_ldmk(ldmk,anchor):
    variances = [0.1,0.2]
    index_x = torch.Tensor([0,2,4,6,8]).long()
    index_y = torch.Tensor([1,3,5,7,9]).long()
    ldmk[:,index_x] = ldmk[:,index_x] * variances[0] * anchor[:,2].view(-1,1) + anchor[:,0].view(-1,1)
    ldmk[:,index_y] = ldmk[:,index_y] * variances[0] * anchor[:,3].view(-1,1) + anchor[:,1].view(-1,1)
    return ldmk
    
import os
# list_per = []

import glob, tqdm
class Face_Alignt():
    def __init__(self, use_gpu = False):
        self.pnet, self.onet = PNet(),ONet() 
        self.pnet.load_state_dict(torch.load('%s/Face_Alignt/weight/msos_pnet_rotate.pt'%os.path.dirname(os.path.abspath(__file__)),map_location=lambda storage, loc:storage), strict=False) 
        self.onet.load_state_dict(torch.load('%s/Face_Alignt/weight/msos_onet_rotate.pt'%os.path.dirname(os.path.abspath(__file__)),map_location=lambda storage, loc:storage), strict=False)
        self.onet.float()
        self.pnet.eval()
        self.onet.eval()
        self.use_gpu = use_gpu
        if self.use_gpu:
            torch.cuda.set_device(0)
            self.pnet.cuda()
            self.onet.cuda()
        else:
            torch.set_num_threads(1)
    def align_multi(self, img, limit=None, min_face_size=30.0, thresholds =  [0.3, 0.6, 0.8], nms_thresholds=[0.6, 0.6, 0.6]):
        boxes, faces =self.detect(img)
        return boxes, faces
    def align(self, img, ):
        boxes, faces = self.detect(img)
        if len(faces) > 0:
            return faces[0]
        return None
    def detect(self, file, limit=None, min_face_size=30.0):
        def change(boxes,ldmks, h, w, pad1):
            index_x = torch.LongTensor([0,2,4,6,8])
            index_y = torch.LongTensor([1,3,5,7,9])
            if h <= w:
                boxes[:,1] = boxes[:,1]*w-pad1
                boxes[:,3] = boxes[:,3]*w-pad1
                boxes[:,0] = boxes[:,0]*w
                boxes[:,2] = boxes[:,2]*w  
                ldmks[:,index_x] = ldmks[:,index_x] * w
                ldmks[:,index_y] = ldmks[:,index_y] * w - torch.Tensor([pad1])
            else:
                boxes[:,1] = boxes[:,1]*h
                boxes[:,3] = boxes[:,3]*h
                boxes[:,0] = boxes[:,0]*h-pad1
                boxes[:,2] = boxes[:,2]*h-pad1
                ldmks[:,index_x] = ldmks[:,index_x] * h - torch.Tensor([pad1])
                ldmks[:,index_y] = ldmks[:,index_y] * h 
            return boxes, ldmks

        if  isinstance(file, np.ndarray):
            im = file
        else:
            if isinstance(file, Image.Image):
                im = np.array(file)
            else:
                im = cv2.imread(file)
                
        if im is None:
            print("can not open image:", file)
            return

        # pad img to square
        h, w,_ = im.shape

        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff //2, dim_diff - dim_diff // 2
        pad = ((pad1,pad2),(0,0),(0,0)) if h<=w else ((0,0),(pad1, pad2),(0,0))
        img = np.pad(im, pad,'constant', constant_values=128)
        
        #get img_pyramid
        img_scale, img_size = 0,int((img.shape[0]-1)/32)
        while img_size > 0:
            img_scale += 1
            img_size /= 2
            if img_scale == 6:
                break
        img_size = 32
        img_pyramid = []
        t_boxes,t_probs, t_anchors, t_crops, t_which = None, None, None, None, None
        
        for scale in range(5):
            # print('scale:{0} img_size:{1}'.format(scale, img_size))
            input_img = cv2.resize(img,(img_size, img_size))
            img_pyramid.append(input_img.transpose(2,0,1))
            im_tensor = torch.from_numpy(input_img.transpose(2,0,1)).float()
            if self.use_gpu:
                im_tensor = im_tensor.cuda()
            #get conf and loc(box)
            if self.use_gpu:
                torch.cuda.synchronize()

            loc, conf = self.pnet(torch.unsqueeze(im_tensor,0))
            if self.use_gpu:
                torch.cuda.synchronize()
        
            # print('forward time:{}s'.format(e_t-s_t))        
            loc, conf = loc.detach().cpu(),conf.detach().cpu() 
            loc, conf = loc.data.squeeze(0),F.softmax(conf.squeeze(0))
            boxes, anchor, crop = decode_box(loc,size=img_size)
            which_img = torch.tensor([scale]).long().expand((crop.shape[0],))
            
            #add box into stack
            if scale == 0:
                t_boxes, t_confs, t_anchors, t_crops, t_which = boxes, conf, anchor, crop, which_img
            else:
                t_boxes = torch.cat((t_boxes, boxes),0)
                t_confs = torch.cat((t_confs, conf),0)
                t_anchors = torch.cat((t_anchors, anchor),0)
                t_crops = torch.cat((t_crops, crop),0)
                t_which = torch.cat((t_which, which_img),0)
            img_size *= 2

        #get right boxes and nms
        t_confs[:,0] = 0.6
        max_conf, labels = t_confs.max(1)
        if labels.long().sum().item() is 0:

            return [], []
        ids = labels.nonzero().squeeze(1)
        t_boxes, t_confs, t_anchors, t_crops, t_which = t_boxes[ids], t_confs[ids], t_anchors[ids], t_crops[ids], t_which[ids]
        max_conf = max_conf[ids]
        
        keep = nms(t_boxes, max_conf)
        t_boxes, max_conf, t_anchors, t_crops, t_which = t_boxes[keep], max_conf[keep], t_anchors[keep], t_crops[keep], t_which[keep]

        t_boxes = t_boxes.detach().numpy()
        max_conf = max_conf.detach().numpy()
        
        #get crop and ldmks
        crop_imgs = []
        for i in range(t_boxes.shape[0]):
            img = img_pyramid[t_which[i]]
            crop = t_crops[i].numpy()
            _,h_,w_ = img.shape
            o_x1,o_y1,o_x2,o_y2 = max(crop[0],0),max(crop[1],0),min(crop[2],w_),min(crop[3],h_)
            c_x1 = 0 if crop[0] >=0 else -crop[0]
            c_y1 = 0 if crop[1] >=0 else -crop[1]
            c_x2 = 64 if crop[2] <= w_ else 64 - (crop[2] - w_)
            c_y2 = 64 if crop[3] <= h_ else 64 - (crop[3] - h_)
            crop_img = np.ones((3,64,64))*128
            np.copyto(crop_img[:,c_y1:c_y2,c_x1:c_x2],img[:,o_y1:o_y2,o_x1:o_x2])
            crop_imgs.append(crop_img)
        crop_imgs = torch.from_numpy(np.array(crop_imgs)).float()
        if self.use_gpu:
            crop_imgs = crop_imgs.cuda()
        t_ldmks = self.onet(crop_imgs).detach().cpu()[:,10,:].squeeze(1)
        t_ldmks = decode_ldmk(t_ldmks, t_anchors)
        t_boxes, t_ldmks = change(t_boxes,t_ldmks, h, w, pad1)
        r_faces = []
        r_bboxes = []
        if limit is None:
            num_face = len(t_boxes)
        else:
            num_face = min(len(t_boxes) + 1, limit)
        for i in range(num_face):

            box, prob, ldmk = t_boxes[i], max_conf[i], t_ldmks[i]
            if prob <= 0.87:
                continue
            ldmk_fn = ldmk.reshape(5,2)
            x1 = max(int(box[0]) - 5, 0)
            x2 = min(int(box[2]) + 5, im.shape[1])
            y1 = max(int(box[1])- 5, 0)
            y2 = min(int(box[3]) + 5, im.shape[0])
            if x2-x1 < 50:
                continue
            bbox = [x1, y1, x2, y2, prob]
            r_bboxes.append(bbox)
            face = alignment(im, ldmk_fn)
            # cv2.rectangle(im, (x1,y1),(x2,y2), (255,0,0), 1)
            # cv2.imwrite('a.png',im)  
            r_faces.append(Image.fromarray(face))
        return np.array(r_bboxes), r_faces
# Face_Alignt = Face_Alignt()
# Face_Alignt.align('PQH_0000.png').convert('RGB').save('a.png', "JPEG")