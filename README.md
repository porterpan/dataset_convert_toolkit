<!--
 * @Description: 
 * @Date: 2024-05-13 19:06:07
 * @LastEditTime: 2024-05-17 10:55:02
 * @Author: porter
 * @FilePath: /dataset_convert_toolkit/README.md
-->
# dataset_convert_toolkit

[![Documentation Status]()]()

deep learning dataset convert toolkit

<video width="320" height="240" controls>
    <source src="img/demo.mp4" type="video/mp4">
</video>

## labelme 

labelme annotation tool and convert labels to VOC, launch labelme and prapar labels.txt for masrk label names. and then gui will be display, you can annotation on it.(data_annotated represent marsked label save path)

``` 
eg: traffic light annocation with command, data_annotated can replease with ./datasetsample/traffic_light/label
labelme data_annotated --labels config/labels.txt --nodata --autosave
```

## dataset convert

1. convert labelme marsk label to yolo datasets format with command.

```
python ./dataset_convert_tool.py  --rectjson2yolo --json_dir "./datasetsample/traffic_light_labelme/label/" --output_dir './traffic_light_yolo' --labels "./config/traffic_light_labelme.yaml"
or 
python ./dataset_convert_tool.py  --rectjson2yolo --json_dir "./datasetsample/traffic_light_labelme/label/" --output_dir './traffic_light_yolo' --labels "config/labels.txt"
or only display with(normal datasets should use up 2 command)
python ./dataset_convert_tool.py  --rectjson2yolo --json_dir "datasetsample/traffic_light_labelme/label" --output_dir './traffic_light_yolo'
```

转换效果demo:

![](img/labelmejson2yolo.png)


2. convert VOC label to yolo datasets format with command.


```
python ./dataset_convert_tool.py  --rectvoc2yolo --xml_dir "datasetsample/SafetyHelmet_VOC/Annotations" --output_dir  "./helmet_yolo" --labels "config/labels_safetyhelmet_voc.txt"
```

VOC 格式的头盔数据集转为yolo 格式数据集效果

![](img/voc2yolo_helmet_demo.png)


convert labelme marsk label to voc datasets format with command.(data_dataset_voc represent label converted of VOC dolder)

```
./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt
```

so this code is refrence from https://github.com/labelmeai/labelme/tree/main/examples/bbox_detection

3. COCO to yolo

![coco folder](img/coco_folder_sample.png)

```
python dataset_convert_tool.py --cocotoyolo --cocojson_file 'datasetsample/coco_test/train' --output_dir './coco_test' --labels 'config/coco_test.yaml'
```

coco to yolo 目前没有测，因为手上没有coco数据集


4. convert labelme line convert to culane formate

```
python dataset_convert_tool.py --linejson2culane --line_json_dir 'datasetsample/laneline_test/label' --output_dir './laneline_culane' --labels 'config/laneline.yaml' --sample_dir 'datasetsample/laneline_test/foxcon' --crop_height 0.5
```

### 生成后的效果

- datasetsample/laneline_test 为供转换参考的demo 样本，
- foxcon/ 为训练的样本，
- laneline_culane/laneseg_label_w16/foxcon 为label真值，将参与训练， 
- laneline_culane/list/trainval_gt.txt 为train和val 的样本list, 
- laneline_culane/marsk_disp/foxcon 为可视化的效果。

![](img/labelmeline2laneline_culane_format.png)

> 需要注意，可以用laneline_culane/foxcon/1682315093.575532.lines.txt 作为评分真值车道线抽样点，每一行表示一个车道线的sample点，**sample点第一个值表示车道线属性值，1表示label为1的车道线，2表示label为2的车道线，注意和culane数据集的区别** [ * lines.txt ] 一般只参与评分。

> label 1,2,3,...表示什么在 config/laneline.yaml 文件中定义，一般0为背景，例如：

```
names: ['ll', 'l', 'r', 'rr']
```
则，0 表示背景，1表示ll，2表示l，3表示r，4表示rr


未来将考虑接入：

- KITTI Road  
- BDD100K
- Waymo Open Dataset
- Argoverse 2
- Lyft Level 5