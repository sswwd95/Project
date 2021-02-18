import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

train_dir = "../project/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
test_dir =  "../project/asl-alphabet/asl_alphabet_test/asl_alphabet_test"

#데이터 클래스 만들어주기
class ASL_Images : 
    asl_character:str # 문자형
    images : []

def load_image(file_path):
    image = cv2.resize(cv2.imread(file_path), (64, 64))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_images(path) : 
    image_collection = []
    for folder in os.listdir(path):
        asl_character_folder = os.path.join(train_dir,folder)
        # 경로를 병합하여 새로운 경로 생성
        asl_character = os.path.basename(asl_character_folder)
        # 경로 중 파일명만 얻기(str :문자형)

        image_list = []
        asl_character_images = ASL_Images(asl_character, image_list)
        image_collection.append(asl_character_images)
        for file in os.listdir(asl_character_folder):
            image_path = os.path.join(asl_character_folder,file)
            asl_img = load_image(image_path)
            image_list.append(asl_img)
        
