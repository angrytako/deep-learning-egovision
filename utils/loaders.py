import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import numpy as np
import random as rd

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                # changed the directory 
                model_features = pd.DataFrame(pd.read_pickle(os.path.join(self.dataset_conf[m].data_path,
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                           pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record:EpicVideoRecord, modality='RGB'):
        ##################################################################
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        
        #the list is a list of OFFSETS, as shown by this line idx_untrimmed = record.start_frame + idx
       
        # "the returned array should have size num_clip x num_frames_per_clip" -> this depends of what a record is
        # A record is a loaded file. If a file is a clip, then this is wrong, otherwise if the file is the entire video
        # then the this is the actual number of frames that need to be sampled. I will assume the second option since 
        # they write it like this. Also is a one dimensional array, judging by how it's used
 
        
        tot_num_frames = record.num_frames[modality]
        is_dense_sampling = self.dense_sampling[modality] if self.dense_sampling is not None else None
        num_clips = self.num_clips

        if is_dense_sampling is None: return np.array([i for i in range(0, (tot_num_frames//num_clips)*num_clips)], dtype=np.int16)

        num_frames_per_clip = self.num_frames_per_clip[modality]
        
        return self._get_dense_sample_(num_clips, num_frames_per_clip, tot_num_frames, stride=2) if is_dense_sampling else self._get_uniform_sample_(num_clips, num_frames_per_clip, tot_num_frames)
        
    def _get_uniform_sample_(self, num_clips, num_frames_per_clip, tot_num_frames):
        intervals = np.int16(np.linspace(0, tot_num_frames, num_clips+1))
        frames = np.array([], dtype=np.int16)
        for i, first in enumerate(intervals[0:-1]):
            last = intervals[i+1]
            frames = np.hstack([frames,np.int16(np.linspace(first, last-1, num_frames_per_clip))])
        return frames
    
    def _get_dense_sample_(self, num_clips, num_frames_per_clip, tot_num_frames, stride=2):
        
        frames_per_part = (num_frames_per_clip-1)//2 if num_frames_per_clip % 2 == 1 else (num_frames_per_clip)//2
        frames = np.array([],dtype=np.int16)
        try:
            video = np.int16(np.linspace(0, tot_num_frames-1, tot_num_frames))
            centrals = np.random.choice(video[frames_per_part*stride+1:-frames_per_part*stride-1], num_clips,replace= False)
          
            for i in centrals:
                if num_frames_per_clip % 2 == 1:
                    frames = np.append(frames,np.concatenate([video[i-stride*frames_per_part:i:stride],[i],video[i+stride:i+stride+stride*frames_per_part:stride]]))
                else: 
                    frames = np.append(frames,np.concatenate([video[i-stride*frames_per_part:i:stride],[i],video[i+stride:i+stride+stride*(frames_per_part-1):stride]]))
            logger.info(frames.shape)
            return frames
       
        except:
          intervals = np.int16(np.linspace(0, tot_num_frames, num_clips+1))
          for i, first in enumerate(intervals[0:-1]):
            last = intervals[i+1]
            frames = np.hstack([frames,np.int16(np.linspace(first, last-1, num_frames_per_clip))])
          return frames
         

    def _get_val_indices(self, record:EpicVideoRecord, modality):
        ##################################################################
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        tot_num_frames = record.num_frames[modality]
        is_dense_sampling = self.dense_sampling[modality] if self.dense_sampling is not None else None
        num_clips = self.num_clips
        if is_dense_sampling is None: return np.array([i for i in range(0, (tot_num_frames//num_clips)*num_clips)], dtype=np.int16)
        
        num_frames_per_clip = self.num_frames_per_clip[modality]
        return self._get_dense_sample_(num_clips, num_frames_per_clip, tot_num_frames, stride=2) if is_dense_sampling else self._get_uniform_sample_(num_clips, num_frames_per_clip, tot_num_frames)
        

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]
        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = images
        if self.transform is not None and self.transform[modality] is not None:
            process_data = self.transform[modality](images)
        else: process_data = np.array(images)
        return process_data, record.label

    def _load_data(self, modality, record:EpicVideoRecord, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        if modality == 'EMG':
            emg_readings_series = pd.read_pickle(os.path.join(data_path, record.untrimmed_video_name+".pkl"))
            return [emg_readings_series[record.uid][idx]]
        
        if modality == 'EMG_SPEC':
            emg_spec_series = pd.read_pickle(os.path.join(data_path, record.untrimmed_video_name+"SPEC.pkl"))
            return [emg_spec_series[record.uid][:,:,idx]]

        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)
