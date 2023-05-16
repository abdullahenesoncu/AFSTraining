#!/bin/bash

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install -r requirements.txt

# Download celeba_hq_256
export fileid='1q-lwjTszcNEFpCEAyknKwegrYEWvoibk';
export filename='celeba_hq_256.zip'
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
     
unzip -o celeba_hq_256.zip
rm celeba_hq_256.zip

# Download StyleGan2 Weights
export fileid='1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT';
export filename='weights/stylegan2.pt'
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
     
# Download e4e Weights
export fileid='1cUv_reLE6k3604or78EranS7XzuVMWeO';
export filename='weights/e4e.pt'
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
     
rm cookies.txt
rm confirm.txt
     
# Download face parsing weights
wget https://github.com/truongvu2000nd/AFS/releases/download/v1.0/face_parsing.pth -O weights/face_parsing.pth

# Download original face parser model
wget https://github.com/truongvu2000nd/AFS/releases/download/v1.0/style_extraction.pth -O weights/style_extraction_original.pth

mkdir results
python3 prepare.py
