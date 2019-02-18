import cv2
from PIL import Image
from pathlib import Path
import torch
from config import get_config
from api import face_recognize
from utils.utils import draw_box_name
import glob
import argparse
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="face recognition")
    parser.add_argument('-image',type=str,help="-image path image")
    parser.add_argument('-path',type=str,help="-path path folder list image")
    parser.add_argument('-threshold',type=float,help="-threshold threshold", default=1.34)
    args = parser.parse_args()
    conf = get_config()
    face_recognize = face_recognize(conf)

    face_recognize._raw_load_single_face(args.image)
    targets = face_recognize.embeddings
    submiter = [['image','x1','y1','x2','y2','result']]
    for img in tqdm(glob.glob(args.path + '/*')):
        temp = [img.split('/')[-1], 0,0,0,0,0]
        image = Image.open(img)
        try:
            bboxes, faces = face_recognize.align_multi(image)
        except:
            bboxes = []
            faces = []
        if len(bboxes) > 0:
            bboxes = bboxes[:,:-1] 
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] 
            results, score = face_recognize.infer(faces, targets)

            for id,(re, sc) in enumerate(zip(results, score)):
                if re != -1:
                    temp = [img.split('/')[-1], bboxes[id][0], bboxes[id][1], bboxes[id][2], bboxes[id][3], 1]
        submiter.append(temp)
    df = pd.DataFrame.from_records(submiter)
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    df = df.sort_values(by=['result'], ascending=False)
    df.to_csv("submit.csv",index=None)