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
from progressbar import ProgressBar

from detect_tool.labelmejson2yolo import *
from utils.file import check_folder

class ARGs:
    def __init__(self):
        parser = argparse.ArgumentParser(description='labelmejson2yolo')
        rectangle = parser.add_argument_group('rectangle datasets', 'rectangle label sample convert to other format')
        covertWK = rectangle.add_mutually_exclusive_group(required=True)
        covertWK.add_argument('--rectjson2yolo', action='store_true')
        covertWK.add_argument('--voc2yolo', action='store_false')
        covertWK.add_argument('--json2voc', action='store_false')
        covertWK.add_argument('--coco2voc', action='store_false')
        covertWK.add_argument('--json2coco', action='store_false')
        covertWK.add_argument('--voc2coco', action='store_false')

        json2yolo = parser.add_argument_group('json2yolo', 'labelme detect label convert to yolo description')
        json2yolo.add_argument('--json_dir', type=str, default='data/sample.json', help='path to labelme json file')        
        json2yolo.add_argument("--labels", help="labels file of yaml format", required=True)
        json2yolo.add_argument('--test_ratio', type=float, default=0.3, help='test ratio')
        json2yolo.add_argument('--random_seed', type=int, default=42, help='random seed for data shuffling')
        
        genneralSet = parser.add_argument_group('genneral', 'general parameter setting')        
        genneralSet.add_argument('--output_dir', type=str, default='output/test_single_output', help='path to output directory')

        
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
        check_folder(args.output_dir)
        convert = LabelmeRectangelJson2yolo(args)
        convert.convert()
    else:
        print("please check you parameter")
    







