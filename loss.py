from helpers import getFaceRepresentations, generateFromLatenSpaceCodes, getPerceptions
from constants import coeffs
from metrics import saveResults
import torch
from torch.nn import CosineSimilarity
cosineSimilarity = CosineSimilarity( dim=1 )

class MSELoss(torch.nn.Module):
   def __init__( self ):
      super(MSELoss, self).__init__()
      self.criterion = torch.nn.MSELoss()

   def forward(self, input, target):
      loss = []
      for i, t in zip(input, target):
         loss.append( self.criterion(i, t).unsqueeze( 0 ) )
      return torch.cat( loss )

mseLoss = MSELoss()

def identityLoss( generatedImages, srcImages  ):
   faceRepresentations = getFaceRepresentations( torch.cat( ( generatedImages, srcImages ) ) )
   generatedFaceRepresentations = faceRepresentations[ : len( generatedImages ) ]
   identityFaceRepresentations = faceRepresentations[ len( generatedImages ) : ]
   return 1 - cosineSimilarity( generatedFaceRepresentations, identityFaceRepresentations )
   
def featureMapLoss( wFeatureMaps, wtFeatureMaps ):
   return mseLoss( wFeatureMaps, wtFeatureMaps )

def perceptionLoss( wGeneratedImages, wtGeneratedImages ):
   perceptions = getPerceptions( torch.cat( ( wGeneratedImages, wtGeneratedImages ) ) )
   wPerceptions = perceptions[ : wGeneratedImages.shape[ 0 ] ]
   wtPerceptions = perceptions[ wGeneratedImages.shape[ 0 ] : ]
   return mseLoss( wPerceptions, wtPerceptions )

def consistencyLoss( wt_sty, w_sty ):
   return mseLoss( wt_sty, w_sty )
   
def computeLoss( w, wt, wt_sty, w_sty, srcImages, targetImages, writer, epoch ):
   generatedImages, featureMaps = generateFromLatenSpaceCodes( torch.cat( ( w, wt ) ) )
   wGeneratedImages, wFeatureMaps = generatedImages[ : w.shape[ 0 ] ], featureMaps[ : w.shape[ 0 ] ]
   wtGeneratedImages, wtFeatureMaps = generatedImages[ w.shape[ 0 ] : ], featureMaps[ w.shape[ 0 ] : ]
   
   idLoss = identityLoss( wGeneratedImages, srcImages )
   featMapLoss = featureMapLoss( wFeatureMaps, wtFeatureMaps )
   percLoss = perceptionLoss( wGeneratedImages, wtGeneratedImages )
   consLoss = consistencyLoss( wt_sty, w_sty )
   
   if epoch % 10 == 9:
      saveResults( srcImages, targetImages, wGeneratedImages, epoch )
   
   '''
   print( 'id', idLoss.mean() )
   print( 'featMap', featMapLoss.mean() )
   print( 'perc', percLoss.mean() )
   print( 'cons', consLoss.mean() )
   print( idLoss.shape, featMapLoss.shape, percLoss.shape, consLoss.shape )
   '''
   
   print( f'idLoss: {idLoss.mean().item()}, featMapLoss: {featMapLoss.mean().item()}, percLoss: {percLoss.mean().item()}, consLoss: {consLoss.mean().item()}' )
   
   writer.add_scalar("idLoss/train", idLoss.mean(), epoch )
   writer.add_scalar("featureMapLoss/train", featMapLoss.mean(), epoch )
   writer.add_scalar("perceptionLoss/train", percLoss.mean(), epoch )
   writer.add_scalar("consistencyLoss/train", consLoss.mean(), epoch )
   
   totalLoss = coeffs[ 0 ] * idLoss + coeffs[ 1 ] * featMapLoss + coeffs[ 2 ] * percLoss + coeffs[ 3 ] * consLoss
   return totalLoss.mean()