import torch
import faiss
import os
import pathlib
from PIL import Image
from helper.feature_extraction import MyVGG16, MyResnet50, RGBHistogram, LBP, MyViT
from helper.dataloader import get_transformation
import streamlit as st
from argparse import ArgumentParser
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cpu')

def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists():
            image_list.append(image_path)
    image_list = sorted(image_list, key=lambda x: x.name)
    return image_list

def retrieve_image(img, feature_extractor, feature_root):
    if (feature_extractor == 'vgg16'):
        extractor = MyVGG16('cpu')
    elif (feature_extractor == 'resnet50'):
        extractor = MyResnet50(device)
    elif (feature_extractor == 'rgbhistogram'):
        extractor = RGBHistogram(device)
    elif (feature_extractor == 'lbp'):
        extractor = LBP(device)
    elif (feature_extractor == 'vit'):
        extractor = MyViT(device)
    transform = get_transformation()

    img = img.convert('RGB')
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(feature_root + '/' + feature_extractor + '.index.bin')

    _, indices = indexer.search(feat, k=11)

    return indices[0]
    
def precision(res_path, name):
    name = name[:-7]
    true = 0
    true_total = 0
    true_arr = []
    for res_name in res_path:
        true_total += 1
        if res_name.find(name) != -1:
            true += 1
        true_arr.append(true / true_total)
    return true_arr
            
def recall(data_path, res_path, query_name):
    query_name = query_name[:-7]
    true_total = 0
    for t in data_path:
        img_name = t.name[:-7]
        if img_name == query_name:
            true_total += 1
    print("Number of true image in database:",true_total)
    true = 0
    true_arr = []
    for res_name in res_path:
        if res_name.find(query_name) != -1:
            true += 1
        true_arr.append(true / true_total)
    return true_arr

def average_precision(res_path, name):
    leng = 0
    prec = precision(res_path, name)
    total = 0
    name = name[:-7]
    for i in range(len(res_path)):
        if res_path[i].find(name) != -1:
            leng +=1 
            total += prec[i]
    return total / leng

def main():
    st.title("Image retrieval")
    st.write("Huynh Vo Ngoc Thanh - 21520449")
    st.sidebar.header("Feature")
    feature_option = st.sidebar.selectbox("Choose feature", ("ResNet50", "VGG16"))
    file = st.file_uploader(
        "Choose image to retrieval",
        ["png", "jpg"],
    )
    if file is None:
        st.text("Waiting to upload...")
    else:
        img = Image.open(file)
        st.image(img, caption="Input image", width=500)
        feature_root = 'feature'
        image_root = 'database'
        feature_option = feature_option.lower()
        retriev = retrieve_image(img, feature_option, feature_root)
        image_list = get_image_list(image_root)
    
        res_path = []
        c1, c2 = st.columns(2)
        for i in range(5):
            path = str(image_list[retriev[i]])
            res_path.append(path)
            retrieve_img = Image.open(path)
            c1.image(retrieve_img, width=200)
        for i in range(5, 10):
            path = str(image_list[retriev[i]])
            res_path.append(path)
            retrieve_img = Image.open(path)
            c2.image(retrieve_img, width=200)
        
        prec = precision(res_path, file.name)
        prec_rounded = [round(num, 2) for num in prec]
        re = recall(image_list, res_path, file.name)
        re_rounded = [round(num, 2) for num in re]
        ap = average_precision(res_path, file.name)
        st.text("Precision:")
        st.text(prec_rounded)
        st.text("Recall:")
        st.text(re_rounded)
        st.text(f"AP@10= {ap:.2f}")        
            
if __name__ == '__main__':
    main()
    
#streamlit run BT2/main.py

#for i in range(10):
#           path = str(image_list[retriev[i]])
#            res_path.append(path)
#            retrieve_img = Image.open(path)
#            st.image(retrieve_img, width=250)

