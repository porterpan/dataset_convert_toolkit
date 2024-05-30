'''******************************************************************************
* Copyright 2024 The porter pan Authors. All Rights Reserved.
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* 
* 	http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*****************************************************************************'''
import os
import random
import yaml
from glob import glob

class filetool:
    @staticmethod  
    def check_folder(folder_name):
        '''
        检查文件夹是否存在, 如果不存在，创建文件夹
        '''
        if not os.path.exists(folder_name):        
            os.makedirs(folder_name, exist_ok=True)
            print(f"文件夹'{folder_name}'已创建。")
        else:
            print(f"文件夹'{folder_name}'已存在。")
        
          
    @staticmethod    
    def generate_sets_val_train_txt(label_file_list, test_ratio=0.2, random_seed=42, tranval_save_dir='./', endwith = '.json', images_dir=''):
        split = ['train', 'val', 'trainval']
        if endwith != '':
            patch_fn_list = [fn for fn in label_file_list if fn.endswith(endwith)]
        else:
            patch_fn_list = label_file_list
        if images_dir != '':
            patch_fn_list = [os.path.join(images_dir, fn) for fn in patch_fn_list]
            
        random.seed(random_seed)
        random.shuffle(patch_fn_list)
        train_num = int((1-test_ratio) * len(patch_fn_list))
        train_patch_list = patch_fn_list[:train_num]
        valid_patch_list = patch_fn_list[train_num:]
        for s in split:
            save_path = os.path.join(tranval_save_dir, s + '.txt')  
            if s == 'train':
                with open(save_path, 'w') as f:
                    for fn in train_patch_list:
                        f.write('%s\n' % fn)
            elif s == 'val':
                with open(save_path, 'w') as f:
                    for fn in valid_patch_list:
                        f.write('%s\n' % fn)
            elif s == 'trainval':
                with open(save_path, 'w') as f:
                    for fn in patch_fn_list:
                        f.write('%s\n' % fn)
        print('\033[32mFinish Producing %s txt file to %s\033[0m' % (s, save_path))
    
        
        
    @staticmethod  
    def readLabelmeRectangelLabelYaml(yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    @staticmethod  
    def readLabelmeRectangelLabelTxt(txt_file):
        with open(txt_file, 'r') as file:
            labels = file.readlines()
            lines = [line.strip() for line in labels]
        return lines
    
    @staticmethod
    def generate_sets_val_train_txt_absolute_paths(images_directory, test_ratio=0.2, random_seed=42, tranval_save_dir='./', endwith = '.json', images_dir=''):
        entries=[]
        for root, dirs, files in os.walk(images_directory):
            for file in files:
                if file.endswith(endwith):
                    entries.append(os.path.join(root, file))
        # entries = os.listdir(images_directory)
        files = [entry for entry in entries if os.path.isfile(os.path.join(images_directory, entry))]
        # absolute_paths = [os.path.abspath(os.path.join(images_directory, file)) for file in files]
        absolute_paths = [os.path.join(images_directory, file) for file in files]
        filetool.generate_sets_val_train_txt(absolute_paths, test_ratio, random_seed, tranval_save_dir, endwith, images_dir)
        for path in absolute_paths:
            print(path)
    
if __name__ == '__main__':
    #  python train_val_split_2.py
    filetool.generate_sets_val_train_txt_absolute_paths('/home/udi/workspace/panchuanchao/yolov8/datasets/traffic_light_yolo/images',
                                                        endwith='.jpg',
                                                        test_ratio=0.2, random_seed=42,
                                                        tranval_save_dir='/home/udi/workspace/panchuanchao/yolov8/datasets/traffic_light_yolo/ImageSets/Main', 
                                                        images_dir='/home/udi/workspace/panchuanchao/yolov8/datasets/traffic_light_yolo/images')
    
    