from AFS.models.e4e import pSp
from AFS.models.stylegan2.stylegan2 import StyleGAN2Generator as Generator
from constants import e4e_weights_path, stylegan2_weights_path, hopenet_weights_path, device, resolution
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch
import argparse
import cv2
import stable_hopenetlite

encoder_ = None
def getEncoder():
   global encoder_
   if encoder_:
      return encoder_
   ckpt = torch.load( e4e_weights_path )
   opts = ckpt[ 'opts' ]
   opts[ 'checkpoint_path' ] = e4e_weights_path
   opts[ 'device' ] = device
   opts = argparse.Namespace( **opts )
   encoder_ = pSp(opts).eval()
   for param in encoder_.parameters():
      param.requires_grad = False
   encoder_ = encoder_.to( device )
   return encoder_

generator_ = None
def getGenerator():
   global generator_
   if generator_:
      return generator_
   generator_ = Generator( truncation=0.65, use_w=True, resolution=1024, device=device ).eval()
   generator_.load_model( stylegan2_weights_path )
   for param in generator_.parameters():
      param.requires_grad = False
   return generator_

vggface2_ = None
def getVggFace2():
   global vggface2_
   if vggface2_:
      return vggface2_
   vggface2_ = InceptionResnetV1( pretrained='vggface2' ).eval().to( device )
   for param in vggface2_.parameters():
      param.requires_grad = False
   return vggface2_

vgg19_ = None
def getVgg19():
   global vgg19_
   if vgg19_:
      return vgg19_
   vgg19_ = models.vgg19( pretrained=True ).to( device )
   for param in vgg19_.parameters():
      param.requires_grad = False
   return vgg19_

hopenet_ = None
def getHopenet():
   global hopenet_
   if hopenet_:
      return hopenet_
   hopenet_ = stable_hopenetlite.shufflenet_v2_x1_0()
   saved_state_dict = torch.load( hopenet_weights_path, map_location=device )
   hopenet_.load_state_dict( saved_state_dict, strict=False )
   hopenet_ = hopenet_.eval().to( device )
   return hopenet_
   
def getImage( path ):
   img = Image.open( path ).convert( 'RGB' ).resize( ( resolution, resolution ) )
   transform = transforms.Compose([
      transforms.PILToTensor(),
   ])
   return transform( img ).to( device ).type( torch.float )

def getFaceRepresentations( images ):
   # n * 3 * h * w
   vggface2 = getVggFace2()
   output = vggface2( ( images - 127.5 ) / 128.0 )
   return output

def getPerceptions( images ):
   # n * 3 * h * w
   vgg19 = getVgg19()
   images = images / 255
   preprocessedImages = transforms.functional.normalize(
      transforms.functional.resize(images, (224, 224)),
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
   )
   output = vgg19( preprocessedImages )
   output_norm = (output - output.min()) / (output.max() - output.min())
   return output_norm

def getLatentSpaceCodes( images ):
   # n * 3 * h * w
   encoder = getEncoder()
   inputBatch = ( ( images - 127.5 ) / 127.5 )
   inputBatch = torch.nn.functional.interpolate( inputBatch, ( 256, 256 ) )
   output = encoder( inputBatch )
   return output

def generateFromLatenSpaceCodes( latentSpaceCodes ):
   # n * n_latent * 512
   generator = getGenerator()
   output, featureMaps = generator.w_plus_forward( latentSpaceCodes, resize=False, output_layers=['convs.5'] )
   featureMap = featureMaps[ 0 ]
   output = output * 127.5 + 127.5
   output = torch.nn.functional.interpolate( output, ( resolution, resolution ) )
   return output, featureMap

def getPoseRepresentations( images ):
   images = images / 255
   preprocessedImages = transforms.functional.normalize(
      transforms.functional.resize(images, (224, 224)),
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
   )
   return getHopenet()( preprocessedImages )
   
def toPILImage( tensor ):
   image = tensor.permute(1, 2, 0).cpu().detach().numpy()[:, :, ::-1]
   cv2.imwrite( '/tmp/aaa.jpg', image )
   return Image.open( '/tmp/aaa.jpg' )


