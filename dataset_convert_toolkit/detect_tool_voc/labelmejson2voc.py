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
# https://github.com/labelmeai/labelme/tree/main/examples/bbox_detection


# labelme data_annotated --labels labels.txt --nodata --autosave

# ./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt

import os
import random
import json
from tqdm import tqdm
from utils.file import filetool
import xml.etree.ElementTree as ET

class LabelmeJson2voc:
    def __init__(self, args):
        self.randomSeed,self.testRatio = args.random_seed, args.test_ratio
        self.path_of_json_folder, self.outputxmlDir = args.json_dir, args.output_dir
        
        self.images_path = os.path.join(args.output_dir, "JPEGImages")
        self.labels_path = os.path.join(args.output_dir, "Annotations")
        self.sets_main = os.path.join(args.output_dir, "ImageSets/Main")    
        filetool.check_folder(self.images_path)
        filetool.check_folder(self.labels_path)
        filetool.check_folder(self.sets_main)
        # 判断三个参数是否有效，无效则报错，提示输入参数
        if self.path_of_json_folder == '' or self.outputxmlDir =='' or not os.path.isfile(args.labels):
            print('--json_dir: ', self.path_of_json_folder)
            print('--output_dir: ', self.outputxmlDir)
            print('--labels: ', args.labels)
            raise ValueError('paremeter error')          
        
        if args.labels.rsplit('.', 1)[1]=='txt':
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
        self.rectangle_labelme_json_files=[x for x in lableme_rectangle_json_file_list if ".json" in x]

    def convert(self):
        print("labeled sample num: ",len(self.rectangle_labelme_json_files))
        pbar = tqdm(total=100)
        bar_count = 1/len(self.rectangle_labelme_json_files)*100
        for rectangle_json_file in self.rectangle_labelme_json_files:
            inputlabelmejsonfilePath = os.path.join(self.path_of_json_folder, rectangle_json_file)
            outputVocFile = os.path.join(self.labels_path, rectangle_json_file.replace('json','xml'))
            # print("file: ", rectangle_json_file)
            self.rectangelJson2voc(inputlabelmejsonfilePath, outputVocFile)
            pbar.update(bar_count)
        pbar.close()
        filetool.generate_sets_val_train_txt(label_file_list=self.rectangle_labelme_json_files, test_ratio=self.testRatio, 
                                        random_seed=self.randomSeed, tranval_save_dir=self.sets_main)
        
        print("\033[35mcovert over, please copy all JEPGImages image samples to folder {}\033[0m".format(self.images_path))

    def prettyXml(self, element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行 
        if element: # 判断element是否有子元素 
            if element.text == None or element.text.isspace(): # 如果element的text没有内容 
                element.text = newline + indent * (level + 1)  
            else: 
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1) 
        temp = list(element) # 将elemnt转成list 
        for subelement in temp: 
            if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致 
                subelement.tail = newline + indent * (level + 1) 
            else: # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个 
                subelement.tail = newline + indent * level 
            self.prettyXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作
    
    
    def create_xml_template(self, folder, filename, path, width, height, objects, outputPath):
        
        root = ET.Element("annotation")

        folder_element = ET.SubElement(root, "folder")
        folder_element.text = folder

        filename_element = ET.SubElement(root, "filename")
        filename_element.text = filename # a.jpg

        path_element = ET.SubElement(root, "path")
        path_element.text = path # 

        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        size = ET.SubElement(root, "size")
        width_element = ET.SubElement(size, "width")
        width_element.text = str(width)
        height_element = ET.SubElement(size, "height")
        height_element.text = str(height)
        depth = ET.SubElement(size, "depth")
        depth.text = "3"

        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"

        for object in objects:
            xmin, ymin, xmax, ymax, label, shape_type = object
            obj = ET.SubElement(root, "object")
            name = ET.SubElement(obj, "name")
            name.text = label
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            xmin_element = ET.SubElement(bndbox, "xmin")
            xmin_element.text = str(xmin)
            ymin_element = ET.SubElement(bndbox, "ymin")
            ymin_element.text = str(ymin)
            xmax_element = ET.SubElement(bndbox, "xmax")
            xmax_element.text = str(xmax)
            ymax_element = ET.SubElement(bndbox, "ymax")
            ymax_element.text = str(ymax)

        
        # tree.write(outputPath, encoding="utf-8", xml_declaration=True)
        self.prettyXml(root, '\t', '\n')
        tree = ET.ElementTree(root)
        tree.write(outputPath, encoding="utf-8", xml_declaration=True)
        # return root
        
    def rectangelJson2voc(self, input_json, out_xmlFile):
        data = json.load(open(input_json,encoding="utf-8"))
        width=data["imageWidth"]
        height=data["imageHeight"]
        imagePath=data['imagePath']     
        points = []
        for i in  data["shapes"]:
            [[x1,y1],[x2,y2]]=i['points']
            label = i['label']
            shape_type = i['shape_type']
            points.append([x1,y1,x2,y2, label, shape_type]) 
        print("{}--->{}".format(os.path.basename(input_json), out_xmlFile))           
        self.create_xml_template('JPEGImages', os.path.basename(imagePath), imagePath, width, height, points, out_xmlFile)