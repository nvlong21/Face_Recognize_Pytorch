import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
import os
from config import get_config
from api import face_recognize
from utils.utils import draw_box_name
from datetime import datetime
import numpy as np
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name",default='video.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.3, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    parser.add_argument("-save_unknow", "--save_unknow", help="save unknow person", default=0, type=int)

    args = parser.parse_args()
    conf = get_config(net_size = 'large', net_mode = 'ir_se', threshold = args.threshold, use_mtcnn = 1)
    face_recognize = face_recognize(conf)
    
    if args.update:
        targets, names = face_recognize.update_facebank()
        print('facebank updated')
    else:
        targets, names = face_recognize.load_facebanks()
        print('facebank loaded')
    if (not isinstance(targets, torch.Tensor)) and face_recognize.use_tensor:
        targets, names = face_recognize.update_facebank()

    cap = cv2.VideoCapture(args.file_name)
    
    cap.set(cv2.CAP_PROP_POS_MSEC, args.begin* 1000)

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(str('{}/{}.avi'.format(conf.facebank_path, args.save_name)),
                                   cv2.VideoWriter_fourcc(*'XVID'), int(fps), (1280,720))
    if args.duration != 0:
        i = 0
    j=0
    count_unknow = 0
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:         
            img_bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_bg)
            try:
                bboxes, faces = face_recognize.align_multi(image, thresholds = [0.5, 0.7, 0.8])
                j+=1
            except:
                bboxes = []
                faces = []

            if len(bboxes) != 0:
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice   
                results, score, embs = face_recognize.infer(faces, targets)

                for idx, bbox in enumerate(bboxes):
                    if results[idx] == -1 and args.save_unknow:
                        new_per = "%s/unknow_%s"%(conf.facebank_path, count_unknow)
                        if not os.path.exists(new_per):
                            os.mkdir(new_per)
                        
                        faces[idx].save('%s/%s.jpg'%(new_per, datetime.now().date().strftime('%Y%m%d')))
                        targets = torch.cat((targets, embs[idx].unsqueeze(0)), dim=0)
                        names =np.append(names, 'unknow_%s'%count_unknow)
                        count_unknow+=1

                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx]] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx]], frame)
            video_writer.write(frame)
            cv2.imshow("face_recognize", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
        else:
            break
        if args.duration != 0:
            i += 1
            if i % 25 == 0:
                print('{} second'.format(i // 25))
            if i > 25 * args.duration:
                break        
    cap.release()
    video_writer.release()
    
