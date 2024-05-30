from __future__ import print_function
import argparse
from ua_detrac_tool.ua_detrac_convert_voc import uadetrac_convert_voc
from ua_detrac_tool.ua_detrac_convert_yolo import uadetrac_convert_yolo


class ARGs:
    def __init__(self):
        parser = argparse.ArgumentParser(prog='ua-detrac_convert_tool',description='ua-detrac dataset convert', formatter_class=argparse.RawDescriptionHelpFormatter, 
                                         epilog='eg:\n \
                                         1. python ua_detrac_convert_tool.py --rectxml2voc --image_dir /home/udi/DataSet1/车辆检测数据集/train --xml_dir /home/udi/DataSet1/车辆检测数据集/Train-XML/DETRAC-Train-Annotations-XML --output_dir ./detrac_train \n \
                                              \n \
                                        2. '
                                         )
        rectanglelabel = parser.add_argument_group('rectangle datasets', 'rectangle label sample convert to other format')
        covertWK = rectanglelabel.add_mutually_exclusive_group(required=True)
        covertWK.add_argument('--rectxml2yolo', action='store_true', help="xml rectangle label convert to yolo datasets")
        covertWK.add_argument('--rectxml2voc', action='store_true', help="xml rectangle label convert to voc datasets")
                        
        parser.add_argument('--xml_dir', type=str, default='Annotation/', help='path of ua-detrac xml file DIR')   
        
        genneralSet = parser.add_argument_group('genneral', 'general parameter setting')  
        
        genneralSet.add_argument('--output_dir', type=str, default='output/traffic_light_dataset', help='out the datasets root directory')      
        genneralSet.add_argument('--image_dir', type=str, default='input/traffic_light_dataset', help='the image root directory')
        # genneralSet.add_argument("--labels", default='', help="labels file of txt(labelme --labels labels.txt) format")        
        genneralSet.add_argument("--disp", action='store_true', help="display the marks image") 
        genneralSet.add_argument('--test_ratio', type=float, default=0.3, help='test ratio')
        genneralSet.add_argument('--random_seed', type=int, default=42, help='random seed for data shuffling')
        
        # parser.print_help()
        self.args = parser.parse_args()
    def get_opts(self):
        return self.args


if __name__ == '__main__':
    args = ARGs().get_opts()
    if args.rectxml2yolo:
        uadetrac = uadetrac_convert_yolo(args)
        uadetrac.convert()
    elif args.rectxml2voc:
        uadetrac = uadetrac_convert_voc(args)
        uadetrac.convert()