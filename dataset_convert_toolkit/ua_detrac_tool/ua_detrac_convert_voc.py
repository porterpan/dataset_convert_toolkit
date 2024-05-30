import os 
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
from utils.file import filetool
import cv2

class uadetrac_convert_voc:
    '''
    convert uadetrac xml to voc
    '''
    def __init__(self, args):
        self.randomSeed,self.testRatio = args.random_seed, args.test_ratio
        self.path_of_xml_folder, self.output_dir = args.xml_dir, args.output_dir
        self.image_dir = args.image_dir
        self.labels=[]
        
        self.images_path = os.path.join(args.output_dir, "JPEGImages")
        self.labels_path = os.path.join(args.output_dir, "Annotations")
        self.sets_main = os.path.join(args.output_dir, "ImageSets/Main")    
        filetool.check_folder(self.images_path)
        filetool.check_folder(self.labels_path)
        filetool.check_folder(self.sets_main)
        
        # 判断三个参数是否有效，无效则报错，提示输入参数
        if self.path_of_xml_folder == '' or self.output_dir =='':# or not os.path.isfile(args.labels):
            print('--xml_dir: ', self.path_of_xml_folder)
            print('--output_dir: ', self.output_dir)
            # print('--labels: ', args.labels)
            raise ValueError('paremeter error')          
        self.class_name = {}
        
        lableme_rectangle_xml_file_list=os.listdir(self.path_of_xml_folder)
        self.rectangle_uadetrac_xml_files=[x for x in lableme_rectangle_xml_file_list if ".xml" in x]

    def convert(self):
        print("labeled sample num: ",len(self.rectangle_uadetrac_xml_files))
        xml_file_lists = []
        for rectangle_xml_file in tqdm(self.rectangle_uadetrac_xml_files):
            inputuadetracfilePath = os.path.join(self.path_of_xml_folder, rectangle_xml_file)
            xml_file_list = self.rectangeluaxml2voc(inputuadetracfilePath)
            if len(xml_file_list)>0:
                xml_file_lists.extend(xml_file_list)
        #     pbar.update(bar_count)
        # pbar.close()
        filetool.generate_sets_val_train_txt(label_file_list=xml_file_lists, test_ratio=self.testRatio, 
                                        random_seed=self.randomSeed, tranval_save_dir=self.sets_main, endwith='.jpg')
        # 生成labels.txt
        with open(os.path.join(self.output_dir, 'labels.txt'), 'w', encoding='utf-8') as file:
            # 遍历列表，并将每个元素写入到文件中
            for item in self.labels:
                file.write(item + '\n')
        print("\033[35mcovert over, please copy all JEPGImages image samples to folder {}\033[0m".format(self.images_path))
        
    def rectangeluaxml2voc(self, input_xml):
        xml_file_list=[]
        with open(input_xml,encoding="utf-8") as f:
            xmlparse = ET.parse(f)
            root = xmlparse.getroot()
            if 'sequence' in root.tag:
                floder_name = root.attrib['name']
            xml_save_folder = os.path.join(self.labels_path,floder_name)
            image_save_floder = os.path.join(self.images_path,floder_name)
            if os.path.exists(xml_save_folder):
                pass
            else:
                os.makedirs(xml_save_folder)
                
            if os.path.exists(image_save_floder):
                pass
            else:
                os.makedirs(image_save_floder)
                
            # 需要将floder_name 创建在self.labels_path中
            ignore_bboxes = []
            for child in root:
                if child.tag == 'ignored_region':
                    for box in child:
                        left = round(float(box.attrib['left']))
                        top = round(float(box.attrib['top']))
                        width = round(float(box.attrib['width']))
                        height = round(float(box.attrib['height']))
                        xmin, xmax = left, left+width
                        ymin, ymax = top,top+height
                        xmin, xmax = sorted([xmin, xmax])
                        ymin, ymax = sorted([ymin, ymax])                        
                        ignore_bboxes.append([xmin, ymin, xmax, ymax])
                elif child.tag == 'frame':
                    frame_name = child.attrib['num']
                    xml_file_save_path = os.path.join(xml_save_folder, frame_name+'.xml')
                    object= []
                    for target_list in  child:
                        bbox = []
                        for target in target_list:
                            if target.tag == 'target':
                                bbox = [[box.attrib['left'], box.attrib['top'], box.attrib['width'], box.attrib['height']]  for box in target if box.tag == 'box']
                                _class = [attr.attrib['vehicle_type']  for attr in target if attr.tag == 'attribute']
                                if _class[0] not in self.labels:
                                    self.labels.append(_class[0])
                                left, top, width, height = bbox[0]
                                box = [round(float(left)), round(float(left)+float(width)), round(float(top)), round(float(top)+float(height)), _class[0]]                                
                                object.append(box)
                                # print(box)
                    jpg_file_save_path = os.path.join(os.path.join(self.images_path, floder_name), 'img{:05d}'.format(int(frame_name))+'.jpg')
                    self.create_xml_template(floder_name, frame_name, jpg_file_save_path, 0,0, object, xml_file_save_path)
                    self.copyImagework(os.path.join(os.path.join(self.image_dir, floder_name), 'img{:05d}'.format(int(frame_name))+'.jpg'), jpg_file_save_path, ignore_bboxes, object=object)
                    xml_file_list.append(jpg_file_save_path)
        return xml_file_list
    
    def copyImagework(self, source_path, target_path, ignore_bboxes, object=[]):
        image = cv2.imread(source_path)
        if image is None:
            print("error read image of :{}".format(source_path))
        for ignore_bbox in ignore_bboxes:
            x1, y1, x2, y2 = ignore_bbox
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        if len(object)>3:
            newimage = image.copy()
            for box in object:                
                x1, x2, y1, y2, cls = box
                p1 = (x1, y1)
                p2 = (x2, y2)
                cv2.rectangle(newimage, p1, p2, [0, 0, 255], 2)
                cv2.putText(newimage, cls, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #cv2.imshow('obj Image', newimage)
            #cv2.waitKey(10)  # 等待按键
        cv2.imwrite(target_path, image)
        
            
    def prettyXml(self, element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行 
        if element: # 判断element是否有子元素 
            if element.text == None or element.text.isspace(): # 如果element的text没有内容 
                element.text = newline + indent * (level + 1)  
            else: 
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1) 
        #else: # 此处两行如果把注释去掉，Element的text也会另起一行 
            #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level 
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
            xmin, ymin, xmax, ymax, label = object
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
    
    