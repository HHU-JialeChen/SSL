import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import pickle
from util import *
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import math
import pdb
urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}
def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    img = img.crop((j, i, j + w, i + h))
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
class NewRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale#default=(0.08,1.0)
        self.ratio = ratio
    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            w2 = int(round(math.sqrt(target_area * aspect_ratio)))
            h2 = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w


        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        #print(i,j,h,w)
        i2, j2, h2, w2 = self.get_params(img, self.scale, self.ratio)
        #print(i2, j2, h2, w2)
        #if( math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)):
        i2=max(min(i2+((i+h//2)-(i2+h2//2))*random.uniform(0.8,1.2), img.size[1]-h2), 0)
        j2 = max(min(j2 + ((j + w // 2) - (j2 + w2 // 2)) * random.uniform(0.8, 1.2), img.size[0] - w2), 0)
        return resized_crop(img, i, j, h, w, self.size, self.interpolation),resized_crop(img, i2, j2, h2, w2, self.size, self.interpolation)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        return format_string
def download_coco2014(data, phase):
    # if not os.path.exists(root):
    #     os.makedirs(root)
    # tmpdir = os.path.join(root, 'tmp/')
    # data = os.path.join(root, 'data/')
    # if not os.path.exists(data):
    #     os.makedirs(data)
    # if not os.path.exists(tmpdir):
    #     os.makedirs(tmpdir)
    # if phase == 'train':
    #     filename = 'train2014.zip'
    # elif phase == 'val':
    #     filename = 'val2014.zip'
    # cached_file = os.path.join(tmpdir, filename)
    # print(cached_file)
    # if not os.path.exists(cached_file):
    #     print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
    #     os.chdir(tmpdir)
    #     subprocess.call('wget ' + urls[phase + '_img'], shell=True)
    #     os.chdir(root)
    # # extract file
    # img_data = os.path.join(data, filename.split('.')[0])
    # if not os.path.exists(img_data):
    #     print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
    #     command = 'unzip {} -d {}'.format(cached_file,data)
    #     os.system(command)
    # print('[dataset] Done!')

    # train/val images/annotations
    # cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    # if not os.path.exists(cached_file):
    #     print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
    #     os.chdir(tmpdir)
    #     subprocess.Popen('wget ' + urls['annotations'], shell=True)
    #     os.chdir(root)
    # annotations_data = os.path.join(data, 'annotations')
    # if not os.path.exists(annotations_data):
    #     print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
    #     command = 'unzip {} -d {}'.format(cached_file, data)
    #     os.system(command)
    # print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(os.path.join(data, 'annotations'), 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')
def download_coco2017(data, phase):
    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(os.path.join(data, 'annotations'), 'instances_{}2017.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')
def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(data.Dataset):
    def __init__(self, root, transform=None, phase='train'): 
        #, inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        
        self.crop=transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC)
        self.newCrop=NewRandomResizedCrop(224, interpolation=Image.BICUBIC)

        self.transform = transforms.Compose([
        
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),  # gao si lv bo
            Solarization(p=0.0),  # guo du bao guang
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        '''
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name
        '''
    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        #new_img_list=[]
        #for img in self.img_list:
        #    for i in range(4):
        #        new_img_list.append(img)
        #self.img_list=new_img_list
        print(self.phase + ' number: ', len(self.img_list))
        self.cat2idx = json.load(open(os.path.join(self.root,  'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, '{}2014'.format(self.phase), filename)).convert('RGB')
        # if self.transform is not None:
        #     img = self.transform(img)
        img1_crop,img2_crop=self.newCrop(img)
        y1 = self.transform(img1_crop)
        y2 = self.transform_prime(img2_crop)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (y1,y2, filename), target
        #return (img, filename, self.inp), target
class COCO2017(data.Dataset):
    def __init__(self, root, transform=None, phase='train'): 
        #, inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        #download_coco2017_trainval(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        
        self.crop=transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC)
        self.newCrop=NewRandomResizedCrop(224, interpolation=Image.BICUBIC)
        if (transform!=None):
            self.transform=transform
        else:
            self.transform = transforms.Compose([
            
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),  # gao si lv bo
                Solarization(p=0.0),  # guo du bao guang
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.transform_prime = transforms.Compose([
    
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        '''
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name
        '''
    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        #new_img_list=[]
        #for img in self.img_list:
        #    for i in range(4):
        #        new_img_list.append(img)
        #self.img_list=new_img_list
        print(self.phase + ' number: ', len(self.img_list))
        self.cat2idx = json.load(open(os.path.join(self.root,  'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, '{}2017'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
           img = self.transform(img)
        
        #img1_crop,img2_crop=self.newCrop(img)
        #y1 = self.transform(img1_crop)
        #y2 = self.transform_prime(img2_crop)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        #return (y1,y2, filename), target
        return img, target
class COCO2017All(data.Dataset):
    def __init__(self, root, transform=None, phase='train'): 
        #, inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        #download_coco2017_trainval(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        
        self.crop=transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC)
        self.newCrop=NewRandomResizedCrop(224, interpolation=Image.BICUBIC)
        if (transform!=None):
            self.transform=transform
        else:
            self.transform = transforms.Compose([
            
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),  # gao si lv bo
                Solarization(p=0.0),  # guo du bao guang
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.transform_prime = transforms.Compose([
    
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        '''
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name
        '''
    def get_anno(self):
        list_path = os.path.join(self.root, 'train_anno.json')
        self.img_list = json.load(open(list_path, 'r'))
        for item in self.img_list:
            item['file_name']= "train2017/"+item['file_name']
        list_path = os.path.join(self.root, 'val_anno.json')
        list2=json.load(open(list_path, 'r'))
        for item in list2:
            item['file_name']= "val2017/"+item['file_name']
        self.img_list=self.img_list+list2
        #new_img_list=[]
        #for img in self.img_list:
        #    for i in range(4):
        #        new_img_list.append(img)
        #self.img_list=new_img_list
        print(self.phase + ' number: ', len(self.img_list))
        self.cat2idx = json.load(open(os.path.join(self.root,  'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root,  filename)).convert('RGB')
        if self.transform is not None:
           img = self.transform(img)
        
        #img1_crop,img2_crop=self.newCrop(img)
        #y1 = self.transform(img1_crop)
        #y2 = self.transform_prime(img2_crop)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        #return (y1,y2, filename), target
        return img, target
class COCO2014_fintune(data.Dataset):
    def __init__(self, root, transform=None, phase='train'):
        #, inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        '''
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name
        '''
    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        print(self.phase + ' number: ', len(self.img_list))
        self.cat2idx = json.load(open(os.path.join(self.root,  'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return img, target
        #return (img, filename), target
        #return (img, filename, self.inp), target
class COCO2017_fintune(data.Dataset):
    def __init__(self, root, transform=None, phase='train'):
        #, inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2017(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        '''
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name
        '''
    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        print(self.phase + ' number: ', len(self.img_list))
        self.cat2idx = json.load(open(os.path.join(self.root,  'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, '{}2017'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        target[target==0]=1
        target[target==-1]=0
        return img, target
        #return (img, filename), target
        #return (img, filename, self.inp), target

class COCO2014Test(data.Dataset):
    def __init__(self, root, transform=None, phase='train'):
        #, inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        #download_coco2014(root, phase)
        self.get_anno()
        #self.num_classes = len(self.cat2idx)
        '''
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name
        '''
    def get_anno(self):
        #list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        if self.phase=='train':
           folder = self.root + '/data/train2014'
        else: 
           folder = self.root + '/data/val2014'

        imgs = []
        for f in sorted(os.listdir(folder)):
            if f.endswith('jpg'):
               imgs.append(f)
        self.img_list = imgs
        print(self.phase + ' number: ', len(self.img_list))
        #self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item  #['file_name']
        img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        #target = np.zeros(self.num_classes, np.float32) - 1
        #target[labels] = 1
        return (img, filename) #, target

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

if __name__ == '__main__':
    dataset = COCO2014("/data/data-pub/COCO2014/", phase='test')