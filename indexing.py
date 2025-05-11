import time
import os
import argparse

import faiss
import torch
from torch.utils.data import DataLoader, SequentialSampler

from helper.feature_extraction import MyResnet50, MyVGG16, RGBHistogram, MyViT, LBP
from helper.indexing import get_faiss_indexer
from helper.dataloader import MyDataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Name of dataset path', required=True, type=str)
    parser.add_argument('--feature_extractor', required=True, type=str, default='rgbhistogram')
    parser.add_argument('--device', required=False, type=str, default='cpu')
    parser.add_argument('--batch_size', required=False, type=int, default=64)
    
    print('----------------- Start indexing ----------------- ')
    start = time.time()
    
    args = parser.parse_args()
    device = torch.device(args.device)
    batch_size = args.batch_size
    data_path = args.data_path
    image_root = os.path.join(data_path)
    feature_root = os.path.join(data_path, 'feature')
    
    os.makedirs(feature_root, exist_ok=True)
    
    if args.feature_extractor == 'resnet50':
        extractor = MyResnet50(device)
    elif args.feature_extractor == 'vgg16':
        extractor = MyVGG16(device)
    elif args.feature_extractor == 'rgbhistogram':
        extractor = RGBHistogram(device)
    elif args.feature_extractor == 'vit':
        extractor = MyViT(device)
    elif args.feature_extractor == 'lbp':
        extractor = LBP(device)
    else:
        print("No matching model found!")
        return

    dataset = MyDataLoader(image_root)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    indexer = get_faiss_indexer(extractor.shape)
    
    for images, image_paths in dataloader:
        image = images.to(device)
        features = extractor.extract_features(image)
        indexer.add(features)
        
    faiss.write_index(indexer, feature_root + '/' + args.feature_extractor + '.index.bin')
    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()