# Training of Face Swapping as A Simple Arithmetic Operation
This repository contains training scripts for [Face Swapping as A Simple Arithmetic Operation](https://arxiv.org/abs/2211.10812).

## Preperation
You need to set `cuda=True` if on a windows pc with cuda.
You need to set `mps=True` if on a mac pc with mps backend(M1+ macs).
To prepare environment, it is enough to run `bash prepare.sh`.
To train, it is enough to run `python3 train.sh`
To generate report, set number of images in `generateReport.py` and run `python3 generateReport.py`

### Requirements ( Mainly )

Look at requirements.txt for more details.

* Python
* pytorch
* Pillow
* opencv-python
* face-alignment
* facenet-pytorch
* matplotlib
* numpy
* tensorboard

### Pretrained Models

* [StyleGAN2 model](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)
* [e4e FFHQ inversion](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view)
* [Face parser model](https://github.com/truongvu2000nd/AFS/releases/download/v1.0/face_parsing.pth)
* [Original style extraction network](https://github.com/truongvu2000nd/AFS/releases/download/v1.0/style_extraction.pth)
* [HopenetLight model](https://github.com/stevenyangyj/deep-head-pose-lite/tree/master/model/shuff_epoch_120.pkl)
* [My style extraction network](https://github.com/abdullahenesoncu/AFS/models/styleExtractor.pth)

## Caveats
* I used 256x256 version of celeba_hq to train faster. One can set resolution at constants.py to any value and use celeba_hq.
* I used vgg_face from InceptionResnetV1 instead of arcface because cannot find suitable pytorch implementation of arcface.
* In the original repo, face segmentation is used to keep background better, I couldn't implement that part.
* Multiprocessing is not allowed for now.

## Examples
You can see examples under `examples`. In an image, first part is source image, second part is target image, third part is my result, and the last part is result from afs official repo.
