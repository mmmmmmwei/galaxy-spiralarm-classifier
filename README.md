# galaxy-spiralarm-classifier
The structural information of spiral galaxies such as the spiral arm number, offer valuable insights into the formation processes of spirals and their physical roles in galaxy evolution. We developed classifiers based on convolutional neural networks (CNNs) to categorise spiral galaxies by their number of spiral arms. A selected dataset from Galaxy Zoo 2 is used for training and evaluation.

# Required packages and environment
Refer installedpackagelist.txt

# Data
GZ2 image dataset: https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images

GZ2 catalog: https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/zoo2MainSpecz.csv.gz


# How to use
## Fixed parameters in paper
Refer fixedparam.txt

## Changing parameters in paper
### Training
```
#-----------------
# upgrade & import
#----------------
bLoadModelEval = False # change this to switch between model training / evaluation

bImportModel = False # change to True if want to download model from google drive
```
### Evaluation
```
#-----------------
# upgrade & import
#----------------
bLoadModelEval = True # change this to switch between model training / evaluation

bImportModel = True # change to True if want to download model from google drive
```
#### Load model through google drive
```
if (bImportModel):
    
    !pip install gdown==4.6.0

    import gdown

    if (bZoobot):
        if (bGradCam==False):
            url_model = 'https://drive.google.com/uc?id=[]' #zoobot
        else:
            url_model = 'https://drive.google.com/uc?id=[]' #zoobot
    else:
        if (bGradCam==False):
            url_model = 'https://drive.google.com/uc?id=[]' #effM
        else:
            url_model = 'https://drive.google.com/uc?id=[]' #effM

    output_model = '/kaggle/working/' + project_name + '.zip'
    gdown.download(url_model, output_model, quiet=False,use_cookies=True)

    # importing the zipfile module
    from zipfile import ZipFile

    # loading the temp.zip and creating a zip object
    with ZipFile(output_model, 'r') as zObject:

        # Extracting all the members of the zip 
        # into a specific location.
        zObject.extractall(
            path=project_name)

    model_path = '/kaggle/working/' + project_name + '/finetune_minimal'
```
### B0 model
```
#-----------------
# training
#-----------------
bZoobot = True # use B0 / V2M
```
### V2M model
```
#-----------------
# training
#-----------------
bZoobot = False # use B0 / V2M
```
### Original 6 classes
```
#-----------------
# data alternate
#-----------------
NUM_CLASS_INIT = 6 # original number of class

bClassSelect = False # use certain class to carry out training/evaluation

if (bClassSelect):
    class_sel_arr = [1,2] # select certain class

bClassMap = False # combine class
class_map = [0,1,2,3,3,4] # class combination

NUM_CLASS = 6 # number of class after select/combine
```
#### Result (B0 model)
```
Epoch 48/50
49/49 [==============================] - 35s 705ms/step - loss: 0.3253 - accuracy: 0.8838 - val_loss: 0.4345 - val_accuracy: 0.8525
Epoch 49/50
49/49 [==============================] - 34s 687ms/step - loss: 0.3053 - accuracy: 0.8953 - val_loss: 0.4284 - val_accuracy: 0.8525
Epoch 50/50
49/49 [==============================] - 34s 694ms/step - loss: 0.3396 - accuracy: 0.8748 - val_loss: 0.4316 - val_accuracy: 0.8525
```
### Merge class 2 and 3 (index start from 0)
```
#-----------------
# data alternate
#-----------------
NUM_CLASS_INIT = 6 # original number of class

bClassSelect = False # use certain class to carry out training/evaluation

if (bClassSelect):
    class_sel_arr = [1,2] # select certain class

bClassMap = True # combine class
class_map = [0,1,2,2,3,4] # class combination

NUM_CLASS = 5 # number of class after select/combine
```
### Merge class 3 and 4 (index start from 0)
```
#-----------------
# data alternate
#-----------------
NUM_CLASS_INIT = 6 # original number of class

bClassSelect = False # use certain class to carry out training/evaluation

if (bClassSelect):
    class_sel_arr = [1,2] # select certain class

bClassMap = True # combine class
class_map = [0,1,2,3,3,4] # class combination

NUM_CLASS = 5 # number of class after select/combine
```
### Select class 1 and 2
```
#-----------------
# data alternate
#-----------------
NUM_CLASS_INIT = 6 # original number of class

bClassSelect = True # use certain class to carry out training/evaluation

if (bClassSelect):
    class_sel_arr = [1,2] # select certain class

bClassMap = True # combine class
class_map = [0,1,2,3,3,4] # class combination

NUM_CLASS = 2 # number of class after select/combine
```
### XAI (gradcam++, smoothgrad)
(Different training method. Hence, different model is required)
```
#-----------------
# optional features
#-----------------
bGradCam = True # carried out XAI (gradcam++, smoothgrad)
```
