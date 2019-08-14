import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt

from my_utils import *
import pyflow.pyflow as pyflow


class VIDDataset(Dataset):
    def __init__(self, phase, data_dir, list_file, preprocess):
        self.data_dir = data_dir
        self.preprocess = preprocess
        
        self.img_list = []
        if phase == 'train':
            with open(list_file, 'r') as f:
                vid_list = [data_dir + 'train/' + v.split(' ')[0] for v in f.read().splitlines()]
                for v in vid_list:
                    imgs = [v + '/' + img for img in os.listdir(v) if img[-5:] == '.JPEG']
                    pairs = [[imgs[i-5], imgs[i]] for i in range(5, len(imgs))]
                    self.img_list += pairs
                    
        elif phase == 'val':
            with open(list_file, 'r') as f:
                frame_list = [data_dir + 'val/' + v.split(' ')[0] for v in f.read().splitlines()]
                for i, img in enumerate(frame_list):
                    frame_id = int(img.split('/')[-1])
                    if frame_id >= 5:
                        self.img_list.append([frame_list[i-5]+'.JPEG', img+'.JPEG'])
                
        class_ids = ['n02691156', 'n02419796', 'n02131653', 'n02834778',
                     'n01503061', 'n02924116', 'n02958343', 'n02402425',
                     'n02084071', 'n02121808', 'n02503517', 'n02118333',
                     'n02510455', 'n02342885', 'n02374451', 'n02129165',
                     'n01674464', 'n02484322', 'n03790512', 'n02324045',
                     'n02509815', 'n02411705', 'n01726692', 'n02355227',
                     'n02129604', 'n04468005', 'n01662784', 'n04530566',
                     'n02062744', 'n02391049']

        self.cls_map = {id:idx for idx, id in enumerate(class_ids)}

    def __len__(self):
        return len(self.img_list)
                
    def __getitem__(self, index):
        img1 = self.img_list[index][0]
        img2 = self.img_list[index][1]
        label1 = img1.replace('Data', 'Annotations').replace('.JPEG', '.xml')
        label2 = img2.replace('Data', 'Annotations').replace('.JPEG', '.xml')
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        label1 = parse_bbox(label1, self.cls_map)
        label2 = parse_bbox(label2, self.cls_map)
        return self.preprocess(img1, img2, label1, label2)
    
    

class Policy(nn.Module):
    
    def __init__(self, img_size):
        super(Policy, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(24, 1)
        
        self.img_size = img_size

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.fc(x.view(-1, 24))
        return x

    
    def preprocess(self, img1, img2, boxes1, boxes2):
            # Read image and compute difference
            h, w, _ = img1.shape
            img1 = cv2.resize(img1, self.img_size)
            img2 = cv2.resize(img2, self.img_size)
            img1 = img1.astype(float) / 255.
            img2 = img2.astype(float) / 255.
            diff = img2 - img1

            # Compute difference of gradients
            sobelx1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
            sobelx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)
            sobely1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)
            sobely2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)
            gradx_diff = sobelx2 - sobelx1
            grady_diff = sobely2 - sobely1

            # Compute optical flow
            u, v, _ = pyflow.coarse2fine_flow(img1, img2, 0.012, 0.75, 20, 7, 1, 30, 0)
            u = np.expand_dims(u, 2)
            v = np.expand_dims(v, 2)

            # bbox mask
            mask = np.zeros([self.img_size[0], self.img_size[1], 1])
            for box in boxes1:
                x1, y1, x2, y2 = rescale_bbox(box, (h,w), self.img_size)
                mask[y1:y2, x1:x2, :] = 1

            inputs = np.concatenate([mask, diff, gradx_diff, grady_diff, u, v], axis=2).transpose(2, 0, 1)
            label = 0 if len(boxes1) == len(boxes2) else 1
            return torch.FloatTensor(inputs), torch.FloatTensor([label])

        
    def train_model(self, data_dir, train_list, val_list, output_model, epochs=1, batch_size=64):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        self.load_state_dict(torch.load(output_model))

        # Optimizer
        optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

        # Dataset
        dataset = VIDDataset('train', data_dir, train_list, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        pos_weight = torch.FloatTensor([4]).to(device)

        for epoch in range(epochs):
            self.train()
            
            tp = 0.0
            pos_gt = 0.0
            pos_pred = 0.0
            running_loss = 0.0

            for i, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                # Run model
                outputs = self.forward(imgs)

                # Compute loss
                loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()

                labels = labels.data.cpu().numpy()
                preds = (outputs >= 0).float().data.cpu().numpy()

                tp += np.sum(preds * labels)
                pos_gt += np.sum(labels)
                pos_pred += np.sum(preds)

                if i % 100 == 99:
                    precision = tp / (pos_pred + 1e-9)
                    recall = tp / (pos_gt + 1e-9)
                    f1 = 2 * precision * recall / (precision + recall + 1e-9)
                    print('Batch %d: loss=%.2f, precision=%.2f, recall=%.2f, f1=%.2f' % \
                          (i, running_loss/100, precision, recall, f1))
                    
                    tp = 0.0
                    pos_gt = 0.0
                    pos_pred = 0.0
                    running_loss = 0.0

            # Update scheduler
            scheduler.step()
            
            # Save model parameters
            torch.save(self.state_dict(), output_model)           
            
            # Validation
            self.validate_model(data_dir, val_list, output_model, batch_size)

        
    def validate_model(self, data_dir, val_list, model_path, batch_size=64):

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        self.load_state_dict(torch.load(model_path))
        self.eval()

        # Dataset
        dataset = VIDDataset('val', data_dir, val_list, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tp = 0.0
        pos_gt = 0.0
        pos_pred = 0.0

        print('========== Validation ===========')
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Run model
            outputs = self.forward(imgs)

            labels = labels.data.cpu().numpy()
            preds = (outputs >= 0).float().data.cpu().numpy()

            tp += np.sum(preds * labels)
            pos_pred += np.sum(preds)
            pos_gt += np.sum(labels)

            precision = tp / (pos_pred + 1e-9)
            recall = tp / (pos_gt + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            print('batch %d: precision=%.2f, recall=%.2f, f1=%.2f' % (i, precision, recall, f1))
            
            
            
class Policy2(nn.Module):
    
    def __init__(self, img_size):
        super(Policy2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(515, 515, 5, padding=2, groups=515),
            nn.ReLU(inplace=True),
            nn.Conv2d(515, 256, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 5, padding=2, groups=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, groups=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, groups=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, groups=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(256, 1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.img_size = img_size
        
        squeezenet = models.squeezenet1_0(pretrained=True)
        squeezenet.eval()
        self.feature = squeezenet.features.to(self.device)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x.view(-1, 256))
        return x

    
    def preprocess(self, img1, img2, boxes1, boxes2):
            # Read image
            h, w, _ = img1.shape
            img1 = cv2.resize(img1, (520, 520))
            img2 = cv2.resize(img2, (520, 520))
            img1 = img1.astype(float) / 255.
            img2 = img2.astype(float) / 255.

            # Extract deep feature
            ts1 = torch.from_numpy(img1.transpose(2,0,1)).unsqueeze(0).to(self.device).float()
            ts2 = torch.from_numpy(img2.transpose(2,0,1)).unsqueeze(0).to(self.device).float()
            feat1 = self.feature(ts1)
            feat2 = self.feature(ts2)
            diff = (feat2 - feat1)**2
            diff = diff.squeeze().data.cpu()

            # Compute optical flow
            img1 = cv2.resize(img1, self.img_size)
            img2 = cv2.resize(img2, self.img_size)
            u, v, _ = pyflow.coarse2fine_flow(img1, img2, 0.012, 0.75, 20, 7, 1, 30, 0)
            u = np.expand_dims(u, 2)
            v = np.expand_dims(v, 2)

            # bbox mask
            mask = np.zeros([self.img_size[0], self.img_size[1], 1])
            for box in boxes1:
                x1, y1, x2, y2 = rescale_bbox(box, (h,w), self.img_size)
                mask[y1:y2, x1:x2, :] = 1

            inputs = np.concatenate([mask, u, v], axis=2).transpose(2, 0, 1)
            inputs = torch.cat((torch.FloatTensor(inputs), diff), 0)
            label = torch.FloatTensor([0 if len(boxes1) == len(boxes2) else 1])
            return inputs, label

        
    def train_model(self, data_dir, train_list, val_list, output_model, epochs=1, batch_size=64):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        self.load_state_dict(torch.load(output_model))

        # Optimizer
        optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

        # Dataset
        dataset = VIDDataset('train', data_dir, train_list, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        pos_weight = torch.FloatTensor([4]).to(device)

        for epoch in range(epochs):
            self.train()
            
            tp = 0.0
            pos_gt = 0.0
            pos_pred = 0.0
            running_loss = 0.0

            for i, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                # Run model
                outputs = self.forward(imgs)

                # Compute loss
                loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()

                labels = labels.data.cpu().numpy()
                preds = (outputs >= 0).float().data.cpu().numpy()

                tp += np.sum(preds * labels)
                pos_gt += np.sum(labels)
                pos_pred += np.sum(preds)

                if i % 100 == 99:
                    precision = tp / (pos_pred + 1e-9)
                    recall = tp / (pos_gt + 1e-9)
                    f1 = 2 * precision * recall / (precision + recall + 1e-9)
                    print('Batch %d: loss=%.2f, precision=%.2f, recall=%.2f, f1=%.2f' % \
                          (i, running_loss/100, precision, recall, f1))
                    
                    tp = 0.0
                    pos_gt = 0.0
                    pos_pred = 0.0
                    running_loss = 0.0

            # Update scheduler
            scheduler.step()
            
            # Save model parameters
            torch.save(self.state_dict(), output_model)           
            
            # Validation
            self.validate_model(data_dir, val_list, output_model, batch_size)

        
    def validate_model(self, data_dir, val_list, model_path, batch_size=64):

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        self.load_state_dict(torch.load(model_path))
        self.eval()

        # Dataset
        dataset = VIDDataset('val', data_dir, val_list, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tp = 0.0
        pos_gt = 0.0
        pos_pred = 0.0

        print('========== Validation ===========')
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Run model
            outputs = self.forward(imgs)

            labels = labels.data.cpu().numpy()
            preds = (outputs >= 0).float().data.cpu().numpy()

            tp += np.sum(preds * labels)
            pos_pred += np.sum(preds)
            pos_gt += np.sum(labels)

            precision = tp / (pos_pred + 1e-9)
            recall = tp / (pos_gt + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            print('batch %d: precision=%.2f, recall=%.2f, f1=%.2f' % (i, precision, recall, f1))
            
            
            

class Policy3(nn.Module):
    
    def __init__(self, img_size):
        super(Policy3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(64, 1)
        
        self.img_size = img_size

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.fc(x.view(-1, 64))
        return x

    
    def preprocess(self, img1, img2, boxes1, boxes2):
            # Read image
            h, w, _ = img1.shape
            img1 = cv2.resize(img1, self.img_size)
            img2 = cv2.resize(img2, self.img_size)
            img1 = img1.astype(float) / 255.
            img2 = img2.astype(float) / 255.
            diff = np.square(img1 - img2)

            # Compute optical flow
            u, v, img2_warp = pyflow.coarse2fine_flow(img2, img1, 0.012, 0.75, 20, 7, 1, 30, 0)
            u = np.expand_dims(u, 2)
            v = np.expand_dims(v, 2)
            diff_warp = np.square(img2 - img2_warp)

            # bbox mask
            mask = np.zeros([self.img_size[0], self.img_size[1], 1])
            for box in boxes2:
                x1, y1, x2, y2 = rescale_bbox(box, (h,w), self.img_size)
                mask[y1:y2, x1:x2, :] = 1

            inputs = np.concatenate([mask, u, v, img2, diff, diff_warp], axis=2).transpose(2, 0, 1)
            label = 1 if len(boxes1) < len(boxes2) else 0
            return torch.FloatTensor(inputs), torch.FloatTensor([label])

        
    def train_model(self, data_dir, train_list, val_list, output_model, epochs=1, batch_size=64):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        self.load_state_dict(torch.load(output_model))

        # Optimizer
        optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

        # Dataset
        dataset = VIDDataset('train', data_dir, train_list, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        pos_weight = torch.FloatTensor([4]).to(device)

        for epoch in range(epochs):
            self.train()
            
            tp = 0.0
            pos_gt = 0.0
            pos_pred = 0.0
            running_loss = 0.0

            for i, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                # Run model
                outputs = self.forward(imgs)

                # Compute loss
                loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()

                labels = labels.data.cpu().numpy()
                preds = (outputs >= 0).float().data.cpu().numpy()

                tp += np.sum(preds * labels)
                pos_gt += np.sum(labels)
                pos_pred += np.sum(preds)

                if i % 100 == 99:
                    precision = tp / (pos_pred + 1e-9)
                    recall = tp / (pos_gt + 1e-9)
                    f1 = 2 * precision * recall / (precision + recall + 1e-9)
                    print('Batch %d: loss=%.2f, precision=%.2f, recall=%.2f, f1=%.2f' % \
                          (i, running_loss/100, precision, recall, f1))
                    
                    tp = 0.0
                    pos_gt = 0.0
                    pos_pred = 0.0
                    running_loss = 0.0

            # Update scheduler
            scheduler.step()
            
            # Save model parameters
            torch.save(self.state_dict(), output_model)           
            
            # Validation
            self.validate_model(data_dir, val_list, output_model, batch_size)

        
    def validate_model(self, data_dir, val_list, model_path, batch_size=64):

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        self.load_state_dict(torch.load(model_path))
        self.eval()

        # Dataset
        dataset = VIDDataset('val', data_dir, val_list, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tp = 0.0
        pos_gt = 0.0
        pos_pred = 0.0

        print('========== Validation ===========')
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Run model
            outputs = self.forward(imgs)
            
            '''print(outputs.data.cpu().numpy()[0])
            img1 = imgs[0,3:6,:,:].data.cpu().numpy()
            diff = imgs[0,6:9,:,:].data.cpu().numpy()
            img1 = img1.transpose(1,2,0)#[:,:,::-1]
            diff = diff.transpose(1,2,0)#[:,:,::-1]
            diff /= np.max(diff)
            fig, ax = plt.subplots(1)
            ax.imshow(img1)
            plt.show()
            ax.imshow(diff)
            plt.show()'''

            labels = labels.data.cpu().numpy()
            preds = (outputs >= 0).float().data.cpu().numpy()

            tp += np.sum(preds * labels)
            pos_pred += np.sum(preds)
            pos_gt += np.sum(labels)

            precision = tp / (pos_pred + 1e-9)
            recall = tp / (pos_gt + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            print('batch %d: precision=%.2f, recall=%.2f, f1=%.2f' % (i, precision, recall, f1))
