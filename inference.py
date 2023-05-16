from PIL import Image
from hNetwork import HNetworkList
import torch
import torch.optim as optim
from constants import styleExtractor_weights_path, device
from helpers import getImage, generateFromLatenSpaceCodes, toPILImage, getLatentSpaceCodes

styleExtractor = HNetworkList().to( device )

checkpoint = torch.load( styleExtractor_weights_path )
styleExtractor.load_state_dict( checkpoint )
print( 'Model is loaded' )

def inference( srcImageFn, tgtImageFn, resImageFn ):
   srcImage = getImage( srcImageFn ).unsqueeze( 0 )
   tgtImage = getImage( tgtImageFn ).unsqueeze( 0 )
   
   w = getLatentSpaceCodes( torch.cat( [ srcImage, tgtImage ] ) )
   ws = w[ : 1 ]
   
   w_sty = styleExtractor( w )
   ws_sty = w_sty[ : 1 ]
   wt_sty = w_sty[ 1 : ]
   
   generated, _ = generateFromLatenSpaceCodes( wt_sty + ws - ws_sty )
   toPILImage( generated[ 0 ] ).save( resImageFn )