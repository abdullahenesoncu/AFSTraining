import os
import torch
import random
from helpers import getImage, getLatentSpaceCodes
from constants import device, folders, latent_space_codes_path

fns = { 
   folder : [ fn.split('.')[0] for fn in os.listdir( f'{folders[folder]}/images' ) if fn.endswith( '.jpg' ) ]
   for folder in folders
}

def findLatentSpaceCodeOfFile( path, latentSpaceCodePath ):
   image = getImage( path )
   latentSpaceCode = getLatentSpaceCodes( image.unsqueeze( 0 ) )[ 0 ]
   torch.save( latentSpaceCode, latentSpaceCodePath )

def getBatch( batchSize, setName='train' ):
   srcFns = random.choices( fns[ setName ], k=batchSize )
   tarFns = random.choices( fns[ setName ], k=batchSize )
   data = []
   for srcFn, tarFn in zip( srcFns, tarFns ):
      srcImage = getImage( f'{folders[setName]}/images/{srcFn}.jpg' )
      wsPath = os.path.join( latent_space_codes_path, f'{srcFn}.pt' )
      if not os.path.exists( wsPath ):
         findLatentSpaceCodeOfFile( f'{folders[setName]}/images/{srcFn}.jpg', wsPath )
      ws = torch.load( wsPath ).to( device )
      
      tarImage = getImage( f'{folders[setName]}/images/{tarFn}.jpg' )
      wtPath = os.path.join( latent_space_codes_path, f'{tarFn}.pt' )
      if not os.path.exists( wtPath ):
         findLatentSpaceCodeOfFile( f'{folders[setName]}/images/{tarFn}.jpg', wtPath )
      wt = torch.load( wtPath ).to( device )
      
      data.append( ( ( srcImage, ws ), ( tarImage, wt ) ) )
   return data