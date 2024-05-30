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
from tqdm import tqdm
import numpy as np
import os
from utils.file import filetool

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
    filetool.check_folder(self.images_path)
    filetool.check_folder(self.labels_path)
    filetool.check_folder(self.sets_main)
    
    # calss_labels = self.readLabelmeRectangelLabel(args.labels)['names']
    if args.labels == "":
      calss_labels = []
    elif args.labels.rsplit('.', 1)[1]=='txt':
      print("txt")
      calss_labels = filetool.readLabelmeRectangelLabelTxt(args.labels)
    elif args.labels.rsplit('.', 1)[1]=='yaml':
      calss_labels = filetool.readLabelmeRectangelLabelYaml(args.labels)['names']
    else:
      raise ValueError('--labels paremeter error, must end with:',
                       '.txt', '.yaml') 
    self.class_name = {}
    for i in range(len(calss_labels)):
      self.class_name[calss_labels[i]]= i
    print("class_name: ", self.class_name)
    lableme_rectangle_json_file_list=os.listdir(self.path_of_json_folder)
    self.rectangle_json_files=[x for x in lableme_rectangle_json_file_list if ".json" in x]
    
  def convert(self):    
    print("labeled sample num: ",len(self.rectangle_json_files))
    pbar = tqdm(total=100)
    bar_count = 1/len(self.rectangle_json_files)*100
    for rectangle_json_file in self.rectangle_json_files:
      inputLabelmeLabelJson = os.path.join(self.path_of_json_folder, rectangle_json_file)
      outputYoloFile = os.path.join(self.labels_path, rectangle_json_file.replace('json','txt'))
      self.rectangelJson2yolo(inputLabelmeLabelJson, outputYoloFile)
      pbar.update(bar_count)
    pbar.close()
    filetool.generate_sets_val_train_txt(label_file_list=self.rectangle_json_files, test_ratio=self.testRatio, 
                                    random_seed=self.randomSeed, tranval_save_dir=self.sets_main, images_dir="")
    print("\033[35mcovert over, please copy all image samples to folder {}\033[0m".format(self.images_path))
  
  def rectangelJson2yolo(self, input_json, out_yoloFile):
    data = json.load(open(input_json,encoding="utf-8"))
    width=data["imageWidth"]
    height=data["imageHeight"]
    all_line=''
    class_name_yolo = ''
    for i in  data["shapes"]:
        [[x1,y1],[x2,y2]]=i['points']
        x1,x2=x1/width,x2/width
        y1,y2=y1/height,y2/height
        cx=(x1+x2)/2
        cy=(y1+y2)/2
        w=abs(x2-x1)
        h=abs(y2-y1)
        if len(self.class_name)>0:
          class_name_yolo = self.class_name[i['label']]          
        elif i["label"] != '':
          print("\033[0;31;40monly display show, not specific class_name, please check --labels labels.txt{}\033[0m".format(self.images_path))
          class_name_yolo = i["label"]
        else:
          raise ValueError('--labels not specific, should have:',
                       '*.txt', '*.yaml')
        line="%s %.4f %.4f %.4f %.4f\n"%(class_name_yolo, cx, cy, w, h)
        all_line+=line
    fh=open(out_yoloFile,'w',encoding='utf-8')
    fh.write(all_line)
    fh.close()
    
    
class DynamicAccess:
  def __getattr__(self, item):
    args = {'json_dir':'/home/udi/WareHouse/dataset/traffic_light_sample/yc_traffic_sample_annocation/label_day_10_22', 
            'output_dir':'./traffic_light', 
            'labels':'config/laneline.yaml'}
    return args.get(item)
  
if __name__ == '__main__':
  args = DynamicAccess()
  convert = LabelmeRectangelJson2yolo(args)
        
