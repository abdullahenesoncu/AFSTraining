import face_alignment
from skimage import io
from helpers import getImage, toPILImage, generateFromLatenSpaceCodes, getPoseRepresentations
import torch
import json
import os
import random
import numpy as np
from inference import inference
from afsInference import afsInference
from constants import folders

fa = face_alignment.FaceAlignment( face_alignment.LandmarksType._2D, flip_input=False, device='cpu' )
def get2dLandmarks( image ):
   toPILImage( image ).save( '/tmp/2dlandmarks.jpg' )
   input = io.imread('/tmp/2dlandmarks.jpg')
   return torch.tensor( fa.get_landmarks(input)[ 0 ] )

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def get2dLandmarksDiff( image1, image2 ):
   l1 = get2dLandmarks( image1 )
   l2 = get2dLandmarks( image2 )
   return ( l1 - l2 ).norm( dim=1 ).mean()

def getIdentityDiff( image1, image2 ):
   from loss import identityLoss
   return identityLoss( image1.unsqueeze( 0 ), image2.unsqueeze( 0 ) )[ 0 ]

def getPoseRepresentation( image ):
   y, p, r = getPoseRepresentations( image.unsqueeze( 0 ) )
   return torch.tensor( [ torch.argmax( y ), torch.argmax( p ), torch.argmax( r ) ] ) * 1.0

def getPoseDiff( image1, image2 ):
   return ( getPoseRepresentation( image1 ) - getPoseRepresentation( image2 ) ).norm()

def getDiff( srcImage, tgtImage, resImage ):
   diff = {}
   diff[ 'id' ] = getIdentityDiff( srcImage, resImage ).item()
   diff[ 'expr' ] = get2dLandmarksDiff( tgtImage, resImage ).item()
   diff[ 'pose' ] = getPoseDiff( tgtImage, resImage ).item()
   return diff

def saveResults( srcImages, tgtImages, resImages, epoch ):
   os.makedirs( 'results', exist_ok=True )
   n = srcImages.size( 0 )
   for i in range( n ):
      srcImage = srcImages[ i ]
      tgtImage = tgtImages[ i ]
      resImage = resImages[ i ]
      images = []
      images.append( srcImage )
      images.append( tgtImage )
      images.append( resImage )
      toPILImage( srcImage ).save( '/tmp/srcImage.jpg' )
      toPILImage( tgtImage ).save( '/tmp/tgtImage.jpg' )
      afsInference( '/tmp/srcImage.jpg', '/tmp/tgtImage.jpg', '/tmp/resImage.jpg' )
      afsResImage = getImage( '/tmp/resImage.jpg' )
      images.append( afsResImage )
      myRes = getDiff( srcImage, tgtImage, resImage )
      afsRes = getDiff( srcImage, tgtImage, afsResImage )
      res = {}
      res[ 'my' ] = myRes
      res[ 'afs' ] = afsRes
      json.dump( res, open( f'results/{epoch}_{i}.json', 'w' ) )
      toPILImage( torch.cat( images, dim=2 ) ).save( f'results/{epoch}_{i}.jpg' )

def generateReport( nPairs=100, reportDir='report' ):
   os.makedirs( reportDir, exist_ok=True )
   fns = os.listdir( os.path.join( folders[ 'test' ], 'images' ) )
   sourceFns = random.sample( fns, nPairs )
   targetFns = random.sample( fns, nPairs )
   results = []
   for i in range( nPairs ):
      srcFn = os.path.join( folders[ 'test' ], 'images', sourceFns[ i ] )
      tgtFn = os.path.join( folders[ 'test' ], 'images', targetFns[ i ] )
      srcImage = getImage( srcFn )
      tgtImage = getImage( tgtFn )
      inference( srcFn, tgtFn, '/tmp/resImage.jpg' )
      myResImage = getImage( '/tmp/resImage.jpg' )
      afsInference( srcFn, tgtFn, '/tmp/resImage.jpg' )
      afsResImage = getImage( '/tmp/resImage.jpg' )
      images = [ srcImage, tgtImage, myResImage, afsResImage ]
      graph = torch.cat( images, dim=2 )
      toPILImage( graph ).save( f'report/{i}.jpg' )
      myDiff = getDiff( srcImage, tgtImage, myResImage )
      afsDiff = getDiff( srcImage, tgtImage, afsResImage )
      results.append( ( myDiff, afsDiff ) )
      
   res = {}
   res = { 'results': results }
   res[ 'myMeanResults'] = {
      'idLoss': np.mean( [ x[0]['id'] for x in results ] ),
      'exprLoss': np.mean( [ x[0]['expr'] for x in results ] ),
      'poseLoss': np.mean( [ x[0]['pose'] for x in results ] ),
   }
   res[ 'afsMeanResults' ] = {
      'idLoss': np.mean( [ x[1]['id'] for x in results ] ),
      'exprLoss': np.mean( [ x[1]['expr'] for x in results ] ),
      'poseLoss': np.mean( [ x[1]['pose'] for x in results ] ),
   }
   json.dump( res, open( 'report/report.json', 'w' ) )