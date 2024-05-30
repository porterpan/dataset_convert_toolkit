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
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import random
from utils.file import filetool

'''voc xml format
<annotation>
	<folder>JPEGImage</folder>
	<filename>000000.jpg</filename>
	<path>D:\Folder\dataset\标注数据\JPEGImage\000000.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>640</width>
		<height>480</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>circle_red</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>168</xmin>
			<ymin>2</ymin>
			<xmax>355</xmax>
			<ymax>186</ymax>
		</bndbox>
	</object>
	<object>
		...
	</object>
</annotation>
'''
class VocXml2YOLO:
    def __init__(self, args):
        self.randomSeed,self.testRatio = args.random_seed, args.test_ratio
        self.path_of_xml_folder, self.outputDir = args.xml_dir, args.output_dir
        
        self.images_path = os.path.join(args.output_dir, "images")
        self.labels_path = os.path.join(args.output_dir, "labels")
        self.sets_main = os.path.join(args.output_dir, "ImageSets/Main")    
        filetool.check_folder(self.images_path)
        filetool.check_folder(self.labels_path)
        filetool.check_folder(self.sets_main)
        if args.labels.rsplit('.', 1)[1]=='txt':
            print("txt: ", args.labels)
            calss_labels = filetool.readLabelmeRectangelLabelTxt(txt_file=args.labels)
        elif args.labels.rsplit('.', 1)[1]=='yaml':
            calss_labels = filetool.readLabelmeRectangelLabelYaml(args.labels)['names']
        else:
            raise ValueError('--labels paremeter error, must end with:',
                        '.txt', '.yaml') 
        self.class_name = {}
        for i in range(len(calss_labels)):
            self.class_name[calss_labels[i]]= i
        print("class_name: ", self.class_name)
        lableme_rectangle_xml_file_list=os.listdir(self.path_of_xml_folder)
        self.rectangle_xml_files=[x for x in lableme_rectangle_xml_file_list if ".xml" in x]

    def convert(self):
        print("labeled sample num: ",len(self.rectangle_xml_files))
        pbar = tqdm(total=100)
        bar_count = 1/len(self.rectangle_xml_files)*100
        # i=0
        for rectangle_xml_file in self.rectangle_xml_files:
            inputVocLabelXml = os.path.join(self.path_of_xml_folder, rectangle_xml_file)
            outputYoloFile = os.path.join(self.labels_path, rectangle_xml_file.replace('xml','txt'))
            # print("file: ", rectangle_json_file)
            self.rectangelVoc2yolo(inputVocLabelXml, outputYoloFile)
            # pbar.update(i/len(self.rectangle_xml_files)*100)
            pbar.update(bar_count)
            # i +=1
        pbar.close()
        filetool.generate_sets_val_train_txt(label_file_list=self.rectangle_xml_files, test_ratio=self.testRatio, 
                                        random_seed=self.randomSeed, tranval_save_dir=self.sets_main, endwith='.xml')
        print("\033[35mcovert over, please copy all JEPGImages image samples to folder {}\033[0m".format(self.images_path))
    
    def convertYoloFormat(self, size, box):
        '''
        将bbox的左上角点、右下角点坐标的格式, 转换为bbox中心点+bbox的w,h的格式并进行归一化
        '''
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)
        
    def rectangelVoc2yolo(self, input_xml, out_yoloFile):        
        with open(input_xml,encoding="utf-8") as f:
            root = ET.parse(f)
        data = root.getroot()
        size = data.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        all_line=''
        for obj in data.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.class_name.keys() or int(difficult) == 1:
                continue
            cls_id = self.class_name['{}'.format(cls)]
            xmlbox = obj.find('bndbox')
            points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text))
            bb = self.convertYoloFormat((width, height), points)
            line="%s %.4f %.4f %.4f %.4f\n"%(str(cls_id), bb[0], bb[1], bb[2], bb[3])
            all_line+=line
        fh=open(out_yoloFile,'w',encoding='utf-8')
        fh.write(all_line)
        fh.close()
