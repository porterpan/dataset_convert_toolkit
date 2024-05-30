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
import os
import random
from utils.file import filetool


class CocoJson2YOLO:
    def __init__(self, args):
        self.randomSeed,self.testRatio = args.random_seed, args.test_ratio
        self.path_of_cocojson_file, self.outputDir = args.cocojson_file, args.output_dir
        self.save_class_path = os.path.join(args.output_dir,'classes.txt')
        
        self.images_path = os.path.join(args.output_dir, "images")
        self.labels_path = os.path.join(args.output_dir, "labels")
        self.sets_main = os.path.join(args.output_dir, "ImageSets/Main")    
        filetool.check_folder(self.images_path)
        filetool.check_folder(self.labels_path)
        filetool.check_folder(self.sets_main)
        
        if args.labels is not None:
            self.save_class_path = args.labels
        else:
            print("txt: ", self.save_class_path)
            # calss_labels = filetool.readLabelmeRectangelLabelTxt(txt_file=args.labels)
        self.data = json.load(open(self.path_of_cocojson_file, 'r'))
        self.class_name = self.jsonParaseClassName()

    def jsonParaseClassName(self):
        id_map = {}
        with open(self.save_class_path, 'w') as f:
            for i, category in enumerate(self.data['categories']):
                f.write(f"{category['name']}\n")
                id_map[category['id']] = i
        return id_map
    
    def convertBox2YWH(self, size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]
        #round函数确定(xmin, ymin, xmax, ymax)的小数位数
        x = round(x * dw, 6)
        w = round(w * dw, 6)
        y = round(y * dh, 6)
        h = round(h * dh, 6)
        return (x, y, w, h)


    def convert(self):       
        for img in tqdm(self.data['images']):
            filename = img["file_name"]
            img_width = img["width"]
            img_height = img["height"]
            img_id = self.class_name["id"]
            head, tail = os.path.splitext(filename)
            ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
            f_txt = open(os.path.join(self.labels_path, ana_txt_name), 'w')
            for ann in self.data['annotations']:
                if ann['image_id'] == img_id:
                    box = self.convertBox2YWH((img_width, img_height), ann["bbox"])
                    f_txt.write("%s %s %s %s %s\n" % (self.class_name[ann["category_id"]], box[0], box[1], box[2], box[3]))
            f_txt.close()
        filetool.generate_yolosets_val_train_txt(label_file_list=self.rectangle_cocojson_files, test_ratio=self.testRatio, 
                                        random_seed=self.randomSeed, tranval_save_dir=self.sets_main)
        print("\033[35mcovert over, please copy all JEPGImages image samples to folder {}\033[0m".format(self.images_path))
        
