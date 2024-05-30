
import json
import argparse
from glob import glob, iglob
import os


class ARGs:
    def __init__(self):
        parser = argparse.ArgumentParser(prog='labelmejsonimagedelete',description='delete labelme json imageData', formatter_class=argparse.RawDescriptionHelpFormatter, 
                                         epilog='eg:\n \
                                         1. python json_imageData_delete.py --json_folder \'json/\''
                                         )
        parser.add_argument('--json_folder', type=str, default='', help='labelme labeled json floder')  
        
        
        worlk = parser.add_argument_group('labelme json work', 'do work')
        labelWK = worlk.add_mutually_exclusive_group(required=True)
        labelWK.add_argument('--delete_imageData', action='store_true', help="delete imageData from labelme labeled json")
        labelWK.add_argument('--change_labelName', action='store_true', help="change label name")
        
        change_labelName = parser.add_argument_group('change label name', 'change the label name')
        change_labelName.add_argument('--json_currentlabel', type=str, default='', help='labelme labeled json floder')  
        change_labelName.add_argument('--json_destlabel', type=str, default='', help='labelme labeled json floder')  
        # parser.print_help()
        self.args = parser.parse_args()
    def get_opts(self):
        return self.args 
 

def delete_json_imageData(json_file_path):
    # 假设你的JSON文件名为"annotations.json"
    # json_file_path = 'annotations.json'
    # 加载JSON文件  
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 将imageData置为空
    if 'imageData' in data:
        data['imageData'] = None  # 不能使用""及''
        print('delete \'{}\' imageData -> \'{}\''.format(os.path.basename(json_file_path), data['imageData']))
    # 将修改后的数据写回到JSON文件
    # with open(json_file_path, 'w') as f:
    #     json.dump(data, f, indent=4)
    if 'imagePath' in data:
        imageFormat = ['.jpg', '.png', '.bmp', '.jpeg']
        baseName = os.path.basename(json_file_path)
        for imgtype in imageFormat:
            imageName = baseName.replace('.json', imgtype)
            path_file = '../images/'+imageName
            if imgtype in data['imagePath']:                
                if data['imagePath'] != path_file:
                    data['imagePath'] = path_file
                    print('change {} label name imagePath -> {}'.format(os.path.basename(json_file_path), path_file))
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

        
        
def change_json_labelname(json_file_path, current_name, dest_name):
    # 假设你的JSON文件名为"annotations.json"
    # json_file_path = 'annotations.json'
    # 加载JSON文件  
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 将imageData置为空
    if 'shapes' in data:
        for labelshape in data['shapes']:
            if labelshape['label'] == current_name:
                print('change {} label name \'{}\' -> {}'.format(os.path.basename(json_file_path), args.json_currentlabel, args.json_destlabel))
                labelshape['label'] = dest_name
                # 将修改后的数据写回到JSON文件
                with open(json_file_path, 'w') as f:
                    json.dump(data, f, indent=4)
    
                
        
if __name__ == '__main__':
    # 删除imageData 
    # python json_file_tool.py --delete_imageData --json_folder '/home/udi/WareHouse/dataset/traffic_light_sample/yc_traffic_sample_annocation/yc_traffic_light_train_datasets/annocation'
    args = ARGs().get_opts()
    if args.delete_imageData:
        if os.path.exists(args.json_folder):
            json_folder_dir=args.json_folder
            json_file_list = glob(os.path.join(json_folder_dir, '*.json'), recursive=True)
            # print(json_file_list)
            for file in json_file_list:                
                delete_json_imageData(file)
        else:
            print('json floder not exist!')
    elif args.change_labelName:
        if os.path.exists(args.json_folder):
            json_folder_dir=args.json_folder
            json_file_list = glob(os.path.join(json_folder_dir, '*.json'), recursive=True)
            for file in json_file_list:                
                change_json_labelname(file, args.json_currentlabel, args.json_destlabel)
        else:
            print('json floder not exist!')
    else:
        print('parameter error!')
        
        
        
    