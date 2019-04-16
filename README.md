# WarpGAN: Automatic Caricature Generation

By Yichun Shi, Debayan Deb and Anil K. Jain

<a href="http://caricaturize.me"><img src="https://raw.githubusercontent.com/seasonSH/WarpGAN/master/assets/caricatureme.png" width="100%"></a>

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" align="right" width="100"/>

A tensorflow implementation of [WarpGAN](https://arxiv.org/abs/1811.10100), a fully automatic network that can generate caricatures given an input face photo. Besides transferring rich texture styles, WarpGAN learns to automatically predict a set of control points that can warp the photo into a caricature, while preserving identity. We introduce an identity-preserving adversarial loss that aids the discriminator to distinguish between different subjects. Moreover, WarpGAN allows customization of the generated caricatures by controlling the exaggeration extent and the visual styles.

## <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" width="25"/> Tensorflow release
Currently this repo is compatible with Tensorflow r1.9.

## <img src="https://image.flaticon.com/icons/svg/149/149366.svg" width="25"/> News
| Date     | Update |
|----------|--------|
| 2019-04-10 | Testing Code |
| 2019-04-07 | Training Code |
| 2019-04-05 | Initial Code Upload |

## <img src="https://image.flaticon.com/icons/svg/182/182321.svg" width="25"/> Citation

    @article{warpgan,
      title = {WarpGAN: Automatic Caricature Generation},
      author = {Shi, Yichun, Deb, Debayan and Jain, Anil K.},
      booktitle = {CVPR},
      year = {2019}
    }

## <img src="https://image.flaticon.com/icons/svg/1/1383.svg" width="25"/> Usage
**Note:** In this section, we assume that you are always in the directory **`$WARPGAN_ROOT/`**
### Preprocessing
1. Download the original images of [WebCaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm) dataset and unzip them into ``data/WebCaricature/OriginalImages``. Rename the images by running
    ``` Shell
    python data/rename.py
    ```
2. Then, normalize all the faces by running the following code:
    ``` Shell
    python align/align_dataset.py data/landmarks.txt data/webcaricacture_aligned_256 --scale 0.7
    ```
    The command will normalize all the photos and caricatures using the landmark points pre-defined in the WebCaricature protocol (we use only 5 landmarks). Notice that during deployment, we will use MTCNN to detect the face landmarks for images not in the dataset.

### Training
1. Before training, you need to download the [discriminator model](https://drive.google.com/open?id=1hcxr7yCiS8om59deMrRXFBYNJSrKtqNT) to initialize the parameters of the disrcimanator, which is pre-trained as an identity classifier. Unzip the files under ```pretrained/discriminator_casia_256/```.

2. The configuration files for training are saved under ```config/``` folder, where you can define the dataset prefix, training list, model file and other hyper-parameters. Use the following command to run the default training configuration:
    ``` Shell
    python train.py config/default.py
    ```
    The command will create an folder under ```log/default/``` which saves all the checkpoints, test samples and summaries. The model directory is named as the time you start training.

### Testing
* Run the test code in the following format:
    ```Shell
    python test.py /path/to/model/dir /path/to/input/image /prefix/of/output/image
    ```
* For example, if you want to use the pre-trained model, download the model and unzip it into ```pretrained/warpgan_pretrained```. Then, run the following command to generate 5 images for captain marvel of different random styles:
    ```Shell
    python test.py pretrained/warpgan_pretrained \
    data/example/CaptainMarvel.jpg \
    result/CaptainMarvel \
    --num_styles 5
    ```
* You can also change the warping extent by using the ```--scale``` argument. For example, the following command doubles the displacement of the warpping control points:
    ```Shell
    python test.py pretrained/warpgan_pretrained \
    data/example/CaptainMarvel.jpg \
    result/CaptainMarvel \
    --num_styles 5 --scale 2.0
    ```

## <img src="https://image.flaticon.com/icons/svg/48/48541.svg" width="25"/> Pre-trained Model
##### Discriminator Initializaiton: 
[Google Drive](https://drive.google.com/open?id=1hcxr7yCiS8om59deMrRXFBYNJSrKtqNT)

##### WarpGAN: 
[Google Drive](https://drive.google.com/open?id=1XwjMGcYIg2qwEKHsC7uSmZayHvnEFhyg)




