# AdvFish - Underwater Fish Recognition via Deep Adversarial Learning
Code for Paper "Large-scale Underwater Fish Recognition via Deep Adversarial Learning"

## Dependencies
```console
Python=3.6, Tensorflow=1.8, keras=2.3.1, tqdm, Pillow
```

## Dataset downloading and pre-processing

see codes in datasets.py

* Fish4Knowledge:
   - step 1: manually download the dataset from http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/
   - step 2: find the datafoler to fish_image/
   - step 3: set the image_path to fish_image/ when call function get_Fish4Knowledge(image_path, train_ratio) in the following model training scripts
   
* QUTFish:
   - step 1: manually download the dataset from https://wiki.qut.edu.au/display/cyphy/Fish+Dataset
   - step 2: find the data folder to QUT_fish_data/
   - step 3: set the image_path to QUT_fish_data/ when call function get_QUTFish(image_path, train_ratio) in the following model training scripts
   
* WildFish:
   - step 1: manually download the dataset from https://github.com/PeiqinZhuang/WildFish (downlaod the BaiduYun version, the GoogleDrive version is not complete) into 4 folders: WildFish_part1/2/3/4
   - step 2: find the data folder to WildFish_part1/2/3/4
   - step 3: call funciton clean_wildfish() to clean WildFish dataset, some of the images are corrupted
   - step 4: set image_paths the folder list when call get_WildFish(image_pathes, train_ratio) in the following model training scripts
   
The training-test split will be activated the first time the above three functions are called, and the image indexes will be save into different files for later use. So the firt time call get_Fish4Knowledge/get_QUTFish/get_WildFish may take a while for the processing to complete.

## Model training

* Standard training

$  python3  train_models.py --d      Fish4Knowledge  \
                            --m      ResNet50        \
                            --l      ce              \
                            --e      50              \
                            --b      32              \
                            --s      224   

* Training using AMSoftmax loss

$  python3  train_models_center.py --d      Fish4Knowledge  \
                                   --m      ResNet50        \
                                   --e      50              \
                                   --b      32              \
                                   --s      224   
                                   
* Training using center loss

$  python3  train_models_center.py --d      Fish4Knowledge  \
                                   --m      ResNet50        \
                                   --e      50              \
                                   --b      32              \
                                   --s      224   
                                 
* AdvFish training

$  python3  train_models_adv.py --d      Fish4Knowledge  \
                                --m      ResNet50        \
                                --e      50              \
                                --b      32              \
                                --s      224             \
                                --eps    0.005        
                                   
** Arguments
* -d: dataset ('Fish4Knowledge', QUTFish or 'WildFish')
* -m: CNN model (ResNet18', 'ResNet34', 'ResNet50', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4')
* -l: loss ('ce')
* -e: total number of training epochs, default 50
* -b: batch size, default 32
* -s: input size, default 224
* -eps: adversarial perturbation size, default 1.2/255

### Links to extenal resources:
* Fish4Knowledge: http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/
* QUTFish: https://wiki.qut.edu.au/display/cyphy/Fish+Dataset
* WildFish: https://github.com/PeiqinZhuang/WildFish


