# dataset_convert_toolkit
[![Documentation Status]()]()

deep learning dataset convert toolkit

# labelme 

labelme annotation tool and convert labels to VOC, launch labelme and prapar labels.txt for masrk label names. and then gui will be display, you can annotation on it.(data_annotated represent marsked label save path)

``` 
eg: traffic light annocation with command, data_annotated can replease with ./datasetsample/traffic_light/label
labelme data_annotated --labels config/labels.txt --nodata --autosave
```

## dataset convert

1. convert labelme marsk label to yolo datasets format with command.

```
python ./dataset_convert_tool.py  --rectjson2yolo --json_dir "./datasetsample/traffic_light_labelme/label/" --output_dir './tr
affic_light' --labels "./config/traffic_light_labelme.yaml"
or 
python ./dataset_convert_tool.py  --rectjson2yolo --json_dir "./datasetsample/traffic_light_labelme/label/" --output_dir './tr
affic_light' --labels "config/labels.txt"
or only display with(normal datasets should use up 2 command)
python ./dataset_convert_tool.py  --rectjson2yolo --json_dir "datasetsample/traffic_light_labelme/label" --output_dir './traffic_light'
```

2. convert VOC label to yolo datasets format with command.

```
python ./dataset_convert_tool.py  --rectvoc2yolo --xml_dir "datasetsample/SafetyHelmet_VOC/Annotations" --output_dir  "./test" --labels "config/labels_safetyhelmet_voc.txt"
```


convert labelme marsk label to voc datasets format with command.(data_dataset_voc represent label converted of VOC dolder)

```
./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt
```

so this code is refrence from https://github.com/labelmeai/labelme/tree/main/examples/bbox_detection

# lableImg

You can edit the data/predefined_classes.txt to load pre-defined classes.

```
labelimg JPEGImage predefined_classes.txt
```

labelimg tool for (VOC convert to yolo format) Steps (YOLO)
- In data/predefined_classes.txt define the list of classes that will be used for your training.
- Build and launch using the instructions above.
- Right below "Save" button in the toolbar, click "PascalVOC" button to switch to YOLO format.
- You may use Open/OpenDIR to process single or multiple images. When finished with a single image, click save.
- A txt file of YOLO format will be saved in the same folder as your image with same name. A file named "classes.txt" is saved to that folder too. "classes.- txt" defines the list of class names that your YOLO label refers to.

or you can use **dataset_convert_toolkit** with below command

```
python ./dataset_convert_tool.py  --rectvoc2yolo --xml_dir "./datasetsample/traffic_light_Voc/label/" --output_dir './tr
affic_light' --labels "./config/traffic_light_Voc.yaml"
```

