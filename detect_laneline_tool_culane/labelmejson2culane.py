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
import shutil
from glob import glob
import numpy as np
import os
from utils.file import filetool
import cv2

'''
labelme rectange json format example
{
  "version": "5.1.1",
  "flags": {},
  "shapes": [
    {
      "label": "l",
      "points": [
        [
          1475.0,
          607.391304347826
        ],
        [
          0.0,
          1042.1739130434783
        ]
      ],
      "group_id": null,
      "shape_type": "linestrip",
      "flags": {}
    },
    ...
    {
      "label": "ll",
      ...
      "group_id": null,
      "shape_type": "linestrip",
      "flags": {}
    }
  ],
  "imagePath": "../image/1682315089.487194.jpg",
  "imageData": "",
  "imageHeight": 1020,
  "imageWidth": 1920
}
'''
# laneline_color_index = [
#   (225, 225, 0),
#   (225, 0, 0),
#   (0, 255, 0),
#   (0, 0, 255),
#   (255, 255, 255),
#   (0, 255, 255),
#   (0, 147, 255),
#   (249, 150, 200),
#   (215, 23, 249),
#   (249, 154, 21),
#   (219, 249, 21),
#   (21, 181, 249),
#   (173, 204, 235)
  
# ]
class LabelmeJson2Culane:
  def __init__(self, args):
    self.randomSeed,self.testRatio = args.random_seed, args.test_ratio
    self.path_of_line_json_folder, self.outputDir, self.crop_height = args.line_json_dir, args.output_dir, args.crop_height
    if os.path.isdir(args.sample_dir):
      self.culane_sampleDIR = args.sample_dir
    else:
      raise ValueError('--sample_dir paremeter error, floder not exist') 
      
    self.lanelineDatasetsSampleFolderName = os.path.basename(self.culane_sampleDIR) 
    
    self.lanelineDatasetsList_DIR = os.path.join(args.output_dir, "list")
    self.lanelineDatasetsMarskdisplay_DIR = os.path.join(args.output_dir, "marsk_disp")     
    self.lanelineDatasetsSegLabelW16_DIR = os.path.join(args.output_dir, "laneseg_label_w16")  
    
    self.lanelineDatasetsSample_DIR = os.path.join(args.output_dir, self.lanelineDatasetsSampleFolderName)
    self.lanelineDatasetsTestSplit_DIR = os.path.join(args.output_dir, "list/test_split")
    # self.sets_main = os.path.join(args.output_dir, "ImageSets/Main") 
    self.lanelineDatasetsSegLabelW16Sample_DIR = os.path.join(self.lanelineDatasetsSegLabelW16_DIR, self.lanelineDatasetsSampleFolderName)
    self.lanelineDatasetsMarskdisplaySample_DIR =  os.path.join(self.lanelineDatasetsMarskdisplay_DIR, self.lanelineDatasetsSampleFolderName)
    
    filetool.check_folder(self.lanelineDatasetsSample_DIR)
    filetool.check_folder(self.lanelineDatasetsTestSplit_DIR)
    filetool.check_folder(self.lanelineDatasetsSegLabelW16Sample_DIR)
    filetool.check_folder(self.lanelineDatasetsMarskdisplaySample_DIR)
    print('sample base: ', self.lanelineDatasetsSampleFolderName)
    print('create datasets folder: ', self.lanelineDatasetsSample_DIR)
    print('create datasets folder: ', self.lanelineDatasetsTestSplit_DIR)
    print('create datasets folder: ', self.lanelineDatasetsSegLabelW16Sample_DIR)
    print('create datasets folder: ', self.lanelineDatasetsMarskdisplaySample_DIR)
    
    if args.labels is not None:
      if args.labels.rsplit('.', 1)[1]=='yaml':
        yamldata = filetool.readLabelmeRectangelLabelYaml(args.labels)
        calss_labels = yamldata['names']
        self.culane_row_anchor = yamldata['culane_row_anchor'] 
        self.colorlist = tuple(tuple(item) for item in yamldata['colorlist'])
      else:
        raise ValueError('--labels paremeter error, must end with:',
                       '.yaml') 
    else:
      raise ValueError('--labels paremeter error, must end with:',
                       '.yaml') 
      
    self.class_name = {}
    self.class_name_marsk_color = {}
    for i in range(len(calss_labels)):
      self.class_name[calss_labels[i]]= i+1
      if i<len(self.colorlist):
        self.class_name_marsk_color[calss_labels[i]] = self.colorlist[i]
      else:
        self.class_name_marsk_color[calss_labels[i]] = self.colorlist[4]
    print("class_name: ", self.class_name)
    lableme_line_json_file_list = os.listdir(self.path_of_line_json_folder)
    self.laneline_json_files = [x for x in lableme_line_json_file_list if ".json" in x]
    
  def convert(self):    
    self.copysampleimage2Datasets()
    print("labeled sample num: ",len(self.laneline_json_files))
    pbar = tqdm(total=100)
    bar_count = 1/len(self.laneline_json_files)*100
    for laneline_json_file in self.laneline_json_files:
      marskimagename = laneline_json_file.replace('json','png')
      inputLabelmeLabelJson = os.path.join(self.path_of_line_json_folder, laneline_json_file)
      outputMasrkLabelPNG = os.path.join(self.lanelineDatasetsSegLabelW16Sample_DIR, marskimagename)
      outputMasrkDisplayPNG = os.path.join(self.lanelineDatasetsMarskdisplaySample_DIR, marskimagename)
      self.linejson2culane(inputLabelmeLabelJson, outputMasrkLabelPNG, outputMasrkDisplayPNG, self.lanelineDatasetsSample_DIR)
      pbar.update(bar_count)
    pbar.close()
    # generate train_gt.txt
    filetool.generate_sets_val_train_txt(label_file_list=self.laneline_json_files, test_ratio=self.testRatio, 
                                    random_seed=self.randomSeed, tranval_save_dir=self.lanelineDatasetsList_DIR, endwith='.json')
    # 生成train_gt.txt val_gt.txt
    self.generate_train_val_gt()
    self.convert_line_points_to_image()
    print("\033[35mcovert over, please check all image samples to folder {}\033[0m".format(self.lanelineDatasetsSample_DIR))
  
  def generate_train_val_gt(self):
    # read train.txt and val.txt
    filelist = ['train', 'val', 'trainval']
    for file in filelist:
      strlines = ''
      file_path = os.path.join(self.lanelineDatasetsList_DIR, file+".txt")
      with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
          line = line.strip()
          if line is None:
            continue
          laneline_point_txt_path = os.path.join(self.lanelineDatasetsSample_DIR, line+'.lines.txt')
          file_match_list = glob(os.path.join(self.lanelineDatasetsSample_DIR, line+'*[.jpg|.png|.jpeg]')) 
          if len(file_match_list)==1:
            filename = file_match_list[0]
          else:
            raise ValueError('{} floder container other image sample please delete.'.format(self.lanelineDatasetsSample_DIR)) 
          
          lanelinelist=[0 for _ in range(len(self.class_name))]
          list_lane = []
          with open(laneline_point_txt_path, 'r') as fd:
            pointsline = fd.readlines()
            for l in pointsline:
              if l.strip() is not None:
                list_lane.append(l.split(' ')[0])
          for value in list_lane:
            lanelinelist[int(value)-1]=1
          sample_path_ = '/'+self.lanelineDatasetsSampleFolderName + '/' + os.path.basename(filename)
          out_label_path = '/laneseg_label_w16/'+self.lanelineDatasetsSampleFolderName + '/' +line+'.png'
          strlines = strlines+"{} {} {}\n".format(sample_path_, out_label_path, ' '.join(str(num) for num in lanelinelist))         
          
      with open(os.path.join(self.lanelineDatasetsList_DIR, file+'_gt.txt'), 'w') as f_gt:
          f_gt.write('%s\n' % strlines)
  
  def corpimage(self, image):
    if len(image.shape) == 3:
      height, width, channels = image.shape
      start_y = int(height * self.crop_height)
      cropped_image = image[start_y:, :, :]
      return cropped_image, start_y
    elif len(image.shape) == 2:
      height, width = image.shape
      start_y = int(height * self.crop_height)
      cropped_image = image[start_y:, :]
      return cropped_image, start_y
    else:
      raise ValueError("input image empyty") 
        
  def copysampleimage2Datasets(self):
    print("sample image dir: ", self.culane_sampleDIR)
    src_image_list = glob(os.path.join(self.culane_sampleDIR, '*[.jpg|.png|.jpeg]'), recursive=True)
    print("source sample image num: ", len(src_image_list))
    for srcfile in src_image_list:
      if not os.path.isfile(srcfile):
          print("%s not exist!" % (srcfile))
      else:
          fpath, fname = os.path.split(srcfile) 
          culane_src = self.lanelineDatasetsSample_DIR
          if not os.path.exists(culane_src):
              os.makedirs(culane_src) 
          save_path = os.path.join(culane_src, fname)
          if os.path.exists(culane_src + fname):
              print("file exsits, not copy!")
              continue
          if self.crop_height == 0:
            shutil.copy(srcfile, save_path) 
          else:
            image = cv2.imread(srcfile)
            cropped_image, _ = self.corpimage(image)
            cv2.imwrite(save_path, cropped_image)
          print("copy %s -> %s" % (srcfile, str(culane_src + fname)))
    
  def create_lane_line_txt(self, marsklabelPointsimage, lines_file_path):       
    marsklabelPointsimage, crop_height_col = self.corpimage(marsklabelPointsimage)
    lines = ''
    name_label_value = [value for value in self.class_name.values()]
    for lanelabelId in name_label_value:
      line=''
      for h_line in self.culane_row_anchor:
        h_line_data = marsklabelPointsimage[h_line -crop_height_col-1, :]
        w_lineIndex = np.where(h_line_data == lanelabelId)[0]        
        if w_lineIndex.size>0:
          line = line + str(w_lineIndex[0]) + " " + str(h_line -crop_height_col-1) + " "
      if len(line) > 8:
        lines = lines + str(lanelabelId) +" "+ line + "\n"
    with open(lines_file_path, 'w') as f:
      f.write(lines)
      
  def linejson2culane(self, input_json, outputmarsklabel, outputmasrkdisp, outputlinepoints):    
    data = json.load(open(input_json,encoding="utf-8"))
    width=data["imageWidth"]
    height=data["imageHeight"]    
    imgRGBblack = np.zeros((height, width, 3), np.uint8) 
    marskdispimage = cv2.cvtColor(imgRGBblack.copy() , cv2.COLOR_RGBA2BGR)
    imgmarsklabelBlack = np.zeros((height, width), np.uint8) 
    marsklabelimage = imgmarsklabelBlack.copy()
    marsklabelPointsimage = imgmarsklabelBlack.copy()
    all_line=''
    for i in range(len(data['shapes'])):   
      name = data['shapes'][i]['label']
      points = data['shapes'][i]['points']      
      color = self.class_name_marsk_color[name]      
      marskdispimage = cv2.polylines(marskdispimage, [np.array(points, dtype=int)], False, color, 30)
      marsklabelimage = cv2.polylines(marsklabelimage, [np.array(points, dtype=int)], False, self.class_name[name], 30)
      marsklabelPointsimage = cv2.polylines(marsklabelPointsimage, [np.array(points, dtype=int)], False, self.class_name[name], 1)
   
    masrkdisplayimage, _ = self.corpimage(marskdispimage)
    cv2.imwrite(outputmasrkdisp, masrkdisplayimage)  
    masrklabelimage, crop_h = self.corpimage(marsklabelimage)
    cv2.imwrite(outputmarsklabel, masrklabelimage) 
    # create points to note lane line
    jsonlabelName = os.path.basename(input_json)    
    linesSavePath = os.path.join(self.lanelineDatasetsSample_DIR, jsonlabelName.replace('.json', '.lines.txt')) 
    self.create_lane_line_txt(marsklabelPointsimage, linesSavePath)

  def convert_line_points_to_image(self):
    point_lane_file_list = glob(os.path.join(self.lanelineDatasetsSample_DIR, '*[.lines.txt]'))     
    for point_file in point_lane_file_list:
      disp_input_image_path = os.path.join(self.lanelineDatasetsMarskdisplaySample_DIR, os.path.basename(point_file).replace('.lines.txt', '.png'))
      disp_save_file_path = os.path.join(self.lanelineDatasetsMarskdisplaySample_DIR, os.path.basename(point_file).replace('.lines.txt', '.lines.jpg'))
      image = cv2.imread(disp_input_image_path)
      with open(point_file,  'r') as fn:
        for line in fn.readlines():
          line = line.strip()
          if line == '':
            continue
          else:
            line = line.split(' ')            
            x_coords = [line[i] for i in range(1, len(line), 2)]
            y_coords = [line[i] for i in range(2, len(line), 2)]
            points = [[int(x_coords[i]), int(y_coords[i])] for i in range(0, len(x_coords))]
            name = [k for k, v in self.class_name.items() if v == int(line[0])]
            if len(name)<1:
              continue
            image = cv2.polylines(image, [np.array(points, dtype=int)], False, (0, 0, 255), 5)            
            cv2.putText(image, name[0], (int(x_coords[-1]), int(y_coords[-1])),cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_name_marsk_color[name[0]], 2, cv2.LINE_AA)
      cv2.imwrite(disp_save_file_path, image)
    
class DynamicAccess:
  def __getattr__(self, item):
    args = {'line_json_dir':'/home/udi/DataSet2/foxcom_lane/label', 
            'sample_dir': '/home/udi/DataSet2/foxcom_lane/image',
            'output_dir':'./culane_yc', 
            'labels':'config/laneline.yaml'}
    return args.get(item)
  
if __name__ == '__main__':
  args = DynamicAccess()
  convert = LabelmeJson2Culane(args)
  convert.convert()
        
