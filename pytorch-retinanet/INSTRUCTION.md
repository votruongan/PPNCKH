# Pytorch Retinanet:

Based on this [original ICCV 2017 Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf).

## Prerequesites:
This git is implemented and tested with Python 3.6, install below libraries by pip with the exact versions (if provided).

```
pip install cffi
pip install pandas
pip install cython
pip install opencv-python
pip install requests
pip install tqdm
pip install matplotlib
pip install scikit-image
pip install tensoboardX
pip install tensorflow
pip install torch==0.4.0 torchvision
```

## Installation:
Install pycocotools by run below scripts in the terminal:

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
make install
cd ../../
```

Install Non-maximum suppression (NMS) extension:

```
cd pytorch-retinanet/lib
bash build.sh
cd ../
```

## Configuration:
I have pull some important configuration into config.yml , you can modify this file instead of digging up the source code. If there is any error appear while running, please feel free to contact me.

## Training:
The network can be trained using the `train.py` script. Currently, two dataloaders are available: COCO and CSV. For training on coco, use

```
python train.py --dataset coco --coco_path ../coco --depth 50
```

For training using a custom dataset, with annotations in CSV format (see below), use

```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
```

Note that the --csv_val argument is optional, in which case no validation will be performed.

## Visualization

To visualize the network detection, use `visualize.py`:

```
python visualize.py --dataset coco --coco_path ../coco --model <path/to/model.pt>
```
This will visualize bounding boxes on the validation set. To visualise with a CSV dataset, use:

```
python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```

## Model

The retinanet model uses a resnet backbone. You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Acknowledgements

- Significant amounts of code are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)
- Original source code of this git is from the [pytorch retinanet implementation](https://github.com/yhenon/pytorch-retinanet.git), you can check its issues section to find if there is a solution to your errors before open a new issue in this git to save some of your time.
