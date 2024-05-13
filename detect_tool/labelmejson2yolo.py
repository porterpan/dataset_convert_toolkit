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
import json
import yaml
import random
import cv2
import numpy as np
import os
from utils.file import check_folder

'''
labelme rectange json format example
{
  "version": "5.1.1",
  "flags": {},
  "shapes": [
    {
      "label": "red",
      "points": [
        [
          836.3157894736843,
          841.5789473684208
        ],
        [
          858.6842105263157,
          895.5263157894733
        ]
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "red_time",
      ...
    }
  ],
  "imagePath": "../iamge_day/1697686330.5707149506.png",
  "imageData": "",
  "imageHeight": 1020,
  "imageWidth": 1920
}
'''

class LabelmeRectangelJson2yolo:
  def __init__(self, args):
    self.randomSeed,self.testRatio = args.random_seed, args.test_ratio
    self.path_of_json_folder, self.outputDir = args.json_dir, args.output_dir
    
    self.images_path = os.path.join(args.output_dir, "images")
    self.labels_path = os.path.join(args.output_dir, "labels")
    self.sets_main = os.path.join(args.output_dir, "ImageSets/Main")    
    check_folder(self.images_path)
    check_folder(self.labels_path)
    check_folder(self.sets_main)
    
    calss_labels = self.readLabelmeRectangelLabel(args.labels)['names']
    self.class_name = {}
    for i in range(len(calss_labels)):
      self.class_name[calss_labels[i]]= i
    print("class_name: ", self.class_name)
    lableme_rectangle_json_file_list=os.listdir(self.path_of_json_folder)
    self.rectangle_json_files=[x for x in lableme_rectangle_json_file_list if ".json" in x]
    
  def convert(self):
    for rectangle_json_file in self.rectangle_json_files:
      inputLabelmeLabelJson = os.path.join(self.path_of_json_folder, rectangle_json_file)
      outputYoloFile = os.path.join(self.labels_path, rectangle_json_file.replace('json','txt'))
      print("file: ", rectangle_json_file)
      self.rectangelJson2yolo(inputLabelmeLabelJson, outputYoloFile)
    self.generate_val_train_txt()
    print("\033[35mcovert over, please copy all image samples to folder {}\033[0m".format(self.images_path))

  def readLabelmeRectangelLabel(self, yaml_file):
    with open(yaml_file, 'r') as file:
      data = yaml.safe_load(file)
    return data
  
  def rectangelJson2yolo(self, input_json, out_yoloFile):
    data = json.load(open(input_json,encoding="utf-8"))
    width=data["imageWidth"]
    height=data["imageHeight"]
    all_line=''
    for i in  data["shapes"]:
        [[x1,y1],[x2,y2]]=i['points']
        x1,x2=x1/width,x2/width
        y1,y2=y1/height,y2/height
        cx=(x1+x2)/2
        cy=(y1+y2)/2
        w=abs(x2-x1)
        h=abs(y2-y1)
        line="%s %.4f %.4f %.4f %.4f\n"%(self.class_name[i['label']],cx,cy,w,h)
        all_line+=line
    fh=open(out_yoloFile,'w',encoding='utf-8')
    fh.write(all_line)
    fh.close()
    
  def generate_val_train_txt(self):
    split = ['train', 'val', 'trainval']
    patch_fn_list = [fn.split('/')[-1][:-5] for fn in self.rectangle_json_files]
    random.seed(self.randomSeed)
    random.shuffle(patch_fn_list)
    train_num = int((1-self.testRatio) * len(patch_fn_list))
    train_patch_list = patch_fn_list[:train_num]
    valid_patch_list = patch_fn_list[train_num:]
    for s in split:
      save_path = os.path.join(self.sets_main, s + '.txt')  
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
    
class DynamicAccess:
  def __getattr__(self, item):
    args = {'json_dir':'/home/udi/WareHouse/dataset/traffic_light_sample/yc_traffic_sample_annocation/label_day_10_22', 
            'output_dir':'./traffic_light', 
            'labels':'/home/udi/WareHouse/dataset/traffic_light_sample/traffic_light_1.yaml'}
    return args.get(item)
  
  
if __name__ == '__main__':
  args = DynamicAccess()
  convert = LabelmeRectangelJson2yolo(args)
        
