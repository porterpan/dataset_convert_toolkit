'''******************************************************************************
* Copyright 2024 The Unity-Drive Authors. All Rights Reserved.
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

from __future__ import print_function
import argparse

from detect_tool_yolo.labelmejson2yolo import *
from detect_tool_yolo.rectvocxml2yolo import VocXml2YOLO
from detect_tool_yolo.cocojson2yolo import CocoJson2YOLO
from detect_laneline_tool_culane.labelmejson2culane import LabelmeJson2Culane
from detect_tool_voc.labelmejson2voc import LabelmeJson2voc

class ARGs:
    def __init__(self):
        parser = argparse.ArgumentParser(prog='convert_tool',description='dataset_convert_tool', formatter_class=argparse.RawDescriptionHelpFormatter, 
                                         epilog='eg:\n \
                                         1. convert labelme line label to mark and line point:\n \
                                            python dataset_convert_tool.py --linejson2culane --line_json_dir \'datasetsample/laneline_test/label\' --output_dir \'./culane\' --labels \'config/laneline.yaml\' --sample_dir \'datasetsample/laneline_test/foxcon\' --crop_height 0.5 \n \
                                        2. '
                                         )
        rectanglelabel = parser.add_argument_group('rectangle datasets', 'rectangle label sample convert to other format')
        covertWK = rectanglelabel.add_mutually_exclusive_group(required=True)
        covertWK.add_argument('--rectjson2yolo', action='store_true', help="labelme rectangle label convert to yolo datasets")
        covertWK.add_argument('--rectvoc2yolo', action='store_true', help="VOC rectangle label convert to yolo datasets")
        covertWK.add_argument('--rectcoco2yolo', action='store_true', help="COCO rectangle label convert to yolo datasets")
        
        covertWK.add_argument('--labelmejson2voc', action='store_true', help="labelme rectangle label convert to VOC datasets")    
        covertWK.add_argument('--cocojson2voc', action='store_true', help="coco rectangle label convert to VOC datasets")
        
        covertWK.add_argument('--json2coco', action='store_true', help="lableme rectangle label convert to coco datasets")
        covertWK.add_argument('--voc2coco', action='store_true', help="voc rectangle label convert to coco datasets")
        
        covertWK.add_argument('--linejson2culane', action='store_true', help="labelme line label convert to culane datasets")
        # covertWK.add_argument('--linejson2culane', action='store_true', help="labelme line label convert to culane datasets")

        rectjson2yolo = parser.add_argument_group('rectjson2yolo', 'labelme detect label convert to yolo description')
        rectjson2yolo.add_argument('--json_dir', type=str, default='label_dir/', help='path to labelme json file DIR') 
        
        rectvoc2yolo = parser.add_argument_group('rectvoc2yolo', 'rectangle voc label convert to yolo description')
        rectvoc2yolo.add_argument('--xml_dir', type=str, default='Annotation/', help='path to labelme json file DIR')   
        
        rectvoc2yolo = parser.add_argument_group('rectcoco2yolo', 'rectangle coco label convert to yolo description')
        rectvoc2yolo.add_argument('--cocojson_file', type=str, default='annotations/captions_train2017.json', help='the coco json file path')  
        
        labelmejson2culane = parser.add_argument_group('labelmejson2culane', 'lane line label base on labelme convert to culane description')
        labelmejson2culane.add_argument('--line_json_dir', type=str, default='annotations/', help='the labelme line json file path')  
        labelmejson2culane.add_argument('--sample_dir', type=str, default='part1/', help='the labelme line source image of sample floder')  
        labelmejson2culane.add_argument('--crop_height', type=float, default=0, help='image height [0-1]. crop_height will only save [crop_height-1]*height') 
        
        
        genneralSet = parser.add_argument_group('genneral', 'general parameter setting')        
        genneralSet.add_argument('--output_dir', type=str, default='output/traffic_light_dataset', help='out the datasets root directory')
        genneralSet.add_argument("--labels", default='', help="labels file of txt(labelme --labels labels.txt) format")        
        genneralSet.add_argument('--test_ratio', type=float, default=0.3, help='test ratio')
        genneralSet.add_argument('--random_seed', type=int, default=42, help='random seed for data shuffling')
        
        # parser.print_help()
        self.args = parser.parse_args()
    def get_opts(self):
        return self.args 


if __name__ == '__main__':
    '''
    eg1: create yolo datasets from labelme json file
    python ./dataset_convert_tool.py  
    --rectjson2yolo 
    --json_dir "/home/udi/WareHouse/dataset/traffic_light_sample/yc_traffic_sample_annocation/label_day_10_22" 
    --output_dir './traffic_light' 
    --labels "/home/udi/WareHouse/dataset/traffic_light_sample/traffic_light_1.yaml"
    --random_seed 42
    '''
    args = ARGs().get_opts()
    if args.rectjson2yolo:
        convert = LabelmeRectangelJson2yolo(args)
        convert.convert()
    elif args.rectvoc2yolo:
        convert = VocXml2YOLO(args)
        convert.convert()
    elif args.rectcoco2yolo:
        convert = CocoJson2YOLO(args)
        convert.convert()
    elif args.linejson2culane:
        convert = LabelmeJson2Culane(args)
        convert.convert()
    elif args.labelmejson2voc:
        convert = LabelmeJson2voc(args)
        convert.convert()
    else:
        print("please check you parameter")
    







