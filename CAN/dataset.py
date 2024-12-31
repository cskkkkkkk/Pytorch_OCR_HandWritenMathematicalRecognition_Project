import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision
import cv2
import numpy as np
import os
from pathlib import Path
from scipy.ndimage import zoom

def label_transform(text,words,start_type = 'sos',end_type = 'eos',pad_type = '<pad>',max_len = 160):
    text = text.split()
    text = [start_type] + text + [end_type]
    # while len(text)<max_len:
    #     text += [pad_type]
    # print(f"{vocab=}")
    text = [i for i in map(lambda x:words.words_dict[x],text)]
    return text
    # return torch.LongTensor(text)

def img_transform(img,size,ratio = 1):
    #downsample
    new_size = (int(img.shape[1]/ratio), int(img.shape[0]/ratio))
    new_im = cv2.resize(img,new_size, cv2.INTER_LANCZOS4)#先进行下采样
    new_im = cv2.resize(img,tuple(size))#再缩放到需要的大小
    new_im = new_im[:,:,np.newaxis]
    to_tensor = torchvision.transforms.ToTensor()
    return to_tensor(new_im)


class MyDataset(Dataset):
    def __init__(
        self, 
        params,
        dataset_dir, 
        words,
        img_transform=img_transform,
        label_transform = label_transform,
        ratio = 2,
        is_train = True
    ):
        self.params = params
        self.is_train = is_train
        self.words = words
        self.dataset_dir = dataset_dir
        self.img_transform = img_transform # 传入图片预处理
        self.label_transform = label_transform # 传入图片预处理
        self.ratio = ratio#下采样率

        # list all npy files under the directory
        npy_files_list = []
        for file_name in os.listdir(self.dataset_dir):
            if file_name.endswith(".npy"):
                npy_files_list.append(file_name)
        split_dict = {
            "train": (0, 0.9),
            "val": (0.9, 1),
        }
        split_tuple = split_dict["train"] if is_train else split_dict["val"]
        n_files = len(npy_files_list)
        self.npy_files_list = npy_files_list[int(split_tuple[0] * n_files): int(split_tuple[1] * n_files)]
        self.n_files = len(self.npy_files_list)
        print(f"files {self.npy_files_list} are used as {'train' if is_train else 'val'} set")

        # determine the number of samples as delimiters; n files has n+1 delimiters
        self.n_samples_list = [0]
        if len(self.npy_files_list) == 0:
            print(f"No .npy file found under the directory {Path(dataset_dir)}")
            exit(1)
        for file_name in self.npy_files_list:
            temp_images_and_labels: list[dict] = np.load(Path(dataset_dir) / file_name, allow_pickle=True)
            self.n_samples_list.append(self.n_samples_list[-1] + len(temp_images_and_labels))
        del temp_images_and_labels

        # record the current loaded file index in self.npy_file_list
        self.current_file_idx = 0
        # init with the first file
        self.images_and_labels: list[dict] = np.load(Path(dataset_dir) / self.npy_files_list[0], allow_pickle=True)

    def __getitem__(self, idx):
        # WARNING: the codes assume shuffle=False

        # whether to load next .npy file
        if idx >= self.n_samples_list[self.current_file_idx + 1]:
            self.images_and_labels = np.load(Path(self.dataset_dir) / self.npy_files_list[self.current_file_idx + 1], allow_pickle=True)
            self.current_file_idx += 1
        
        idx_in_list = idx - self.n_samples_list[self.current_file_idx]
        image = self.images_and_labels[idx_in_list]["image"]
        if image.shape[-1] == 3:
            # convert RGB to grayscale and normalize
            # (w, h, c) -> (w, h)
            image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]) / 255
        # Perform downsampling using scipy's zoom function
        image = zoom(image, 1 / self.ratio, order=1)  # order=1 corresponds to bilinear interpolation
        image = torch.tensor(image).float()
        label: str = self.images_and_labels[idx_in_list]["label"]
        label_list = self.label_transform(label,self.words)
        image=image.unsqueeze(0)

        # reset the counter and init between epochs
        if idx == self.n_samples_list[-1] - 1:
            self.current_file_idx = 0
            self.images_and_labels = np.load(Path(self.dataset_dir) / self.npy_files_list[0], allow_pickle=True)

        return image, torch.LongTensor(label_list)

    def __len__(self):
        #return self.n_samples_list[-1]
        return 2


class MyDatasetTest(Dataset):
    def __init__(self,dataset_dir,ratio=2):
        self.dataset_dir = dataset_dir
        self.images_and_labels: list[dict] = np.load(Path(dataset_dir) / 'batch_0.npy', allow_pickle=True)
        self.images_and_labels = sorted(self.images_and_labels, key=lambda x: x['ID'])

        self.ratio = ratio
        self.label_transform = label_transform
    def __getitem__(self, idx):
        image=self.images_and_labels[idx]["image"]
        if image.shape[-1] == 3:
            # convert RGB to grayscale and normalize
            # (w, h, c) -> (w, h)
            image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]) / 255
        image = zoom(image, 1 / self.ratio, order=1)
        image = torch.tensor(image).float()
        return image
    def __len__(self):
        return len(self.images_and_labels)




class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        image = torch.Tensor(255-image) / 255
        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        return image, words


# def get_crohme_dataset(params):
#     words = Words(params['word_path'])
#     params['word_num'] = len(words)
#     print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
#     print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

#     train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
#     eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

#     train_sampler = RandomSampler(train_dataset)
#     eval_sampler = RandomSampler(eval_dataset)

#     train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
#                               num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
#     eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
#                               num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

#     print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
#           f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
#     return train_loader, eval_loader
def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    #print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    #print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = MyDataset(params, params['train_image_path'], words, is_train=True)
    eval_dataset = MyDataset(params, params['train_image_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'],
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, 
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


# def collate_fn(batch_images):
#     max_width, max_height, max_length = 0, 0, 0
#     batch, channel = len(batch_images), batch_images[0][0].shape[0]
#     print(f'batch_images: {batch_images[0]}')
#     proper_items = []
#     for item in batch_images:
#         if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
#             continue
#         max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
#         max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
#         max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
#         proper_items.append(item)

#     images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
#     labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

#     for i in range(len(proper_items)):
#         _, h, w = proper_items[i][0].shape
#         images[i][:, :h, :w] = proper_items[i][0]
#         image_masks[i][:, :h, :w] = 1
#         l = proper_items[i][1].shape[0]
#         labels[i][:l] = proper_items[i][1]
#         labels_masks[i][:l] = 1
#     return images, image_masks, labels, labels_masks
def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    #print(f'batch_images: {batch_images[0]}')
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        #[0]代表一张图片，[1]代表标签
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks

class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label

    def decode_test(self, label_index):
        # 假设 0 是 sos，1 是 eos，当然，你可以根据实际的情况调整
        sos_index = 0
        eos_index = 1
        # 过滤掉 sos 和 eos，并拼接剩下的标签
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index 
                      if int(item) != sos_index and int(item) != eos_index])
    
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}
