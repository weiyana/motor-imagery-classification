import numpy as np
import torch
from torch.utils.data import Dataset
from mne.filter import resample
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import Preprocessor as BraindPreprocessor
from braindecode.datautil.preprocess import preprocess
from braindecode.datautil.windowers import create_windows_from_events

from utils.utils import print_off, print_on
from data_loader.preprocessor import Preprocessor
import os.path as osp
import scipy.io as scio
from utils.band_pass import butter_bandpass_filter


class GRASP(Dataset):
    def __init__(self, options, root, split, fs=250):
        self.options = options
        self.fs=fs
        self.X=self.y=self.spatial_feat=None
        self.load_data(root, split, options.subject)
        self.torch_form()
        # self.X, self.y = Preprocessor.label_selection(self.X, self.y, options.labels)
        self.segmentation()
        # self.torch_form()

    def load_data(self, root, split, subject):
        include_rest=False
        channels=[]#15,43
        if isinstance(subject, int):
            subject = [subject]
        # load data
        all_X=[]
        all_y=[]
        all_spatial_feat=[]
        for subject_id in subject:
            split_folder=split[0].upper()+split[1:]
            data_path=osp.join(root,split_folder,'sample{:02d}.mat'.format(subject_id))
            data = scio.loadmat(data_path)

            epo=data['epo'][0][0]
            # fs=epo[1].item()
            x=epo[3]
            if channels:
                x= x[:,channels,:]
            if not include_rest:
                x=x[3*self.fs+1:,...]
            y=epo[4][1]

            # spatial 
            mnt=data['mnt'][0][0]
            c_x=mnt[0]
            c_y=mnt[1]
            c_3d=mnt[2]

            x,y,c_x,c_y,c_3d=[torch.from_numpy(i) for i in [x,y,c_x,c_y,c_3d]]
            spatial_feat=torch.cat([c_x,c_y,c_3d.T],dim=1)
            all_X.append(x.permute(2,1,0))
            all_y.append(y)
            all_spatial_feat.append(spatial_feat)

        self.X=torch.cat(all_X) # trial,channel,time
        self.y=torch.cat(all_y)
        self.spatial_feat=torch.cat(all_spatial_feat)


    def segmentation(self):
        self.X = Preprocessor.segment_tensor(self.X, self.options.window_size, self.options.step)

    def torch_form(self):
        self.X = self.X.type(torch.FloatTensor)
        self.y = self.y.long()
    
    def band_pass(self,X):
        # bands=[(0.1,4),(5,7),(8,13),(14,30)]
        filtered_X_bands=[]
        if self.options.band:
            # for lowcut,highcut in self.options.band:
            #     filtered_X_channel=[]
            #     for X_channel in X:
            #         X_channel_band=butter_bandpass_filter(X_channel, lowcut, highcut, self.fs, order=6)
            #         filtered_X_channel.append(torch.from_numpy(X_channel_band))
            #     filtered_X_channel=torch.stack(filtered_X_channel)
            #     filtered_X.append(filtered_X_channel)
            # filtered_X=torch.stack(filtered_X)
            for lowcut,highcut in self.options.band:
                filtered_X=butter_bandpass_filter(X, lowcut, highcut, self.fs, order=6)     
                filtered_X_bands.append(torch.from_numpy(filtered_X))
            # import ipdb;ipdb.set_trace()
            filtered_X=torch.stack(filtered_X_bands,dim=1)
        else:
            filtered_X=X.unsqueeze(1)
        return filtered_X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # data filter
        # n_seg, n_band, e, t
        X=self.X[idx]
        filtered_X=self.band_pass(X)
        # filtered_X=filtered_X.unsqueeze(0)
        return filtered_X,self.y[idx]#,self.spatial_feat[idx]



