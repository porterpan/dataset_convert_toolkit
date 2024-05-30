import os 
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
from utils.file import filetool
import cv2

class uadetrac_convert_yolo:
    '''
    convert uadetrac xml to yolo
    '''
    def __init__(self, args):
      self.randomSeed,self.testRatio = args.random_seed, args.test_ratio
      self.path_of_xml_folder, self.outputDir = args.xml_dir, args.output_dir
      self.image_dir = args.image_dir
      self.display_flag = args.disp
      
      self.images_path = os.path.join(args.output_dir, "images")
      self.labels_path = os.path.join(args.output_dir, "labels")
      self.sets_main = os.path.join(args.output_dir, "ImageSets/Main")    
      filetool.check_folder(self.images_path)
      filetool.check_folder(self.labels_path)
      filetool.check_folder(self.sets_main)
      
      self.image_height = 540
      self.image_width = 960
      # if args.labels == "":
      self.labels = []
      # elif args.labels.rsplit('.', 1)[1]=='txt':
      #   # print("txt")
      #   calss_labels = filetool.readLabelmeRectangelLabelTxt(args.labels)
      # elif args.labels.rsplit('.', 1)[1]=='yaml':
      #   calss_labels = filetool.readLabelmeRectangelLabelYaml(args.labels)['names']
      # else:
      #   raise ValueError('--labels paremeter error, must end with:',
      #                   '.txt', '.yaml') car



      self.class_name = {"others":0, "car":1, "van":2, "bus":3, "motorcycle":4, "bicycle":5, "pedestrian":6}
      # for i in range(len(calss_labels)):
      #   self.class_name[calss_labels[i]]= i
      print("class_name: ", self.class_name)
        
      lableme_rectangle_xml_file_list=os.listdir(self.path_of_xml_folder)
      self.rectangle_uadetrac_xml_files=[x for x in lableme_rectangle_xml_file_list if ".xml" in x]

    def convert(self):
      print("labeled sample num: ",len(self.rectangle_uadetrac_xml_files))
      yolotxt_file_lists = []
      for rectangle_xml_file in tqdm(self.rectangle_uadetrac_xml_files):
          inputuadetracfilePath = os.path.join(self.path_of_xml_folder, rectangle_xml_file)
          yolotxt_file_list = self.rectangeluaxml2yolo(inputuadetracfilePath)
          if len(yolotxt_file_list)>0:
              yolotxt_file_lists.extend(yolotxt_file_list)
      #     pbar.update(bar_count)
      # pbar.close()
      filetool.generate_sets_val_train_txt(label_file_list=yolotxt_file_lists, test_ratio=self.testRatio, 
                                      random_seed=self.randomSeed, tranval_save_dir=self.sets_main, endwith='.jpg')
      # 生成labels.txt
      # with open(os.path.join(self.output_dir, 'labels.txt'), 'w', encoding='utf-8') as file:
      #     # 遍历列表，并将每个元素写入到文件中
      #     for item in self.labels:
      #         file.write(item + '\n')
      print("\033[35mcovert over, please copy all JEPGImages image samples to folder {}\033[0m".format(self.images_path))
        
    def rectangeluaxml2yolo(self, input_xml):
      yolotxt_file_list=[]
      with open(input_xml,encoding="utf-8") as f:
          xmlparse = ET.parse(f)
          root = xmlparse.getroot()
          if 'sequence' in root.tag:
              floder_name = root.attrib['name']
          yolotxt_save_folder = os.path.join(self.labels_path,floder_name)
          image_save_floder = os.path.join(self.images_path,floder_name)
          
          
          if os.path.exists(yolotxt_save_folder):
              pass
          else:
              os.makedirs(yolotxt_save_folder)
              
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
                  yolotxt_file_save_path = os.path.join(yolotxt_save_folder, 'img{:05d}'.format(int(frame_name))+'.txt')
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
                  # self.create_xml_template(floder_name, frame_name, jpg_file_save_path, 0,0, object, xml_file_save_path)
                  # TODO:: 
                  self.create_yolotxt_template(object, yolotxt_file_save_path)
                  self.copyImagework(os.path.join(os.path.join(self.image_dir, floder_name), 'img{:05d}'.format(int(frame_name))+'.jpg'), jpg_file_save_path, ignore_bboxes, object=object)
                  yolotxt_file_list.append(jpg_file_save_path)
      return yolotxt_file_list
    
    def copyImagework(self, source_path, target_path, ignore_bboxes, object=None):
      image = cv2.imread(source_path)
      for ignore_bbox in ignore_bboxes:
          x1, y1, x2, y2 = ignore_bbox
          image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
      if object is not None:
          newimage = image.copy()
          for box in object:                
              x1, x2, y1, y2, cls = box
              p1 = (x1, y1)
              p2 = (x2, y2)
              cv2.rectangle(newimage, p1, p2, [0, 0, 255], 2)
              cv2.putText(newimage, cls, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
      if self.display_flag:
        cv2.imshow('obj Image', newimage)
        cv2.waitKey(10)  # 等待按键
      cv2.imwrite(target_path, image)
        
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
    
    
    def create_yolotxt_template(self, obbox, save_path):
      all_line=""
      for obbpoint in obbox:
        line=""
        size=[self.image_width, self.image_height]
        yololabelPoint = self.convertYoloFormat(size, obbpoint)
        cls_id = self.class_name[obbpoint[-1]]
        line="%s %.4f %.4f %.4f %.4f\n"%(str(cls_id), yololabelPoint[0], yololabelPoint[1], yololabelPoint[2], yololabelPoint[3])
        all_line+=line
        fh=open(save_path,'w',encoding='utf-8')
        fh.write(all_line)
        fh.close()
      pass
    
    