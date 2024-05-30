# UA-DETRAC 数据集转换说明

## 1. UA-DETRAC  数据集转换为VOC

```bash
python ua_detrac_convert_tool.py --rectxml2voc --image_dir /home/udi/DataSet1/车辆检测数据集/train --xml_dir /home/udi/DataSet1/车辆检测数据集/Train-XML/DETRAC-Train-Annotations-XML --output_dir ./detrac_train
```





## 2. UA-DETRAC 数据集转换为为YOLO 格式数据集


完成上面步骤1的UA-DETRAC数据集格式转换为VOC数据集后，使用下面的转换工具将VOC格式数据集转换为YOLO格式数据集。

```bash
python dataset_convert_toolkit/ua_detrac_convert_tool.py --rectxml2yolo --image_dir /home/udi/workspace/panchuanchao/Dataset/cardetectdataset/Insight-MVT_Annotation_Train --xml_dir /home/udi/workspace/panchuanchao/Dataset/cardetectdataset/DETRAC-Train-Annotations-XML --output_dir ./detrac_train_yolo
```
需要注意这个--rectxml2yolo 转换需要用官方的数据集格式，满足两点：1.图像宽，高(self.image_height = 540, self.image_width = 960), 以及标注的label(self.class_name = {"others":0, "car":1, "van":2, "bus":3, "motorcycle":4, "bicycle":5, "pedestrian":6})

-  注意： 因为数据集中包含子目录分类，目前没有加上这个子目录分类后的转换后生成train.txt 和val.txt等，需要自己转换。yolo 中images的*.jpg 名字需要和*.txt 的名字一致,否则yolo会找不到对应的label名字。数据集名字不能出现中文，否则很容易出问题，yolo8训练中就出现了数据集带中文提示：UnicodeDecodeError: 'ascii' codec can't decode byte 0xe8 in position 41: ordinal not in range(128)

```
├── images
│   ├── MVI_20012
        ├── img00002.jpg
        ├── img00003.jpg
        ├── img00004.jpg
        ├── img00005.jpg
        ├── img00006.jpg
        ├── img00007.jpg
        ├── img00008.jpg
        ├── img00009.jpg
        ├── img00010.jpg
        ├── img00011.jpg
        ├── img00012.jpg
        ├── img00013.jpg
        ├── img00014.jpg
        ....
│   ├── MVI_20032
        ├── img00002.jpg
......
ImageSets
│   └── Main
└── labels
    ├── MVI_20012
    ├── img00001.txt
        ├── img00002.txt
        ├── img00003.txt
        ├── img00004.txt
        ├── img00005.txt
        ├── img00006.txt
        ├── img00007.txt
        ├── img00008.txt
        ├── img00009.txt
        ├── img00010.txt
        ├── img00011.txt
        ├── img00012.txt
        ├── img00013.txt
        ├── img00014.txt
        ....
    ├── MVI_20032
        ├── img00002.txt
        ......
......

```



本案例中使用yolov8 尝试模型的训练和预测，训练和预测的代码在：

```

```

