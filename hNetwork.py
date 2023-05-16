import torch
import torch.nn as nn
from constants import n_latent

class Highway(nn.Module):
   # This module is copied from https://gist.github.com/dpressel/3b4780bafcef14377085544f44183353
   def __init__(self, input_size):
      super(Highway, self).__init__()
      self.proj = nn.Linear(input_size, input_size)
      self.transform = nn.Linear(input_size, input_size)
      self.transform.bias.data.fill_(-2.0)

   def forward(self, input):
      proj_result = nn.functional.relu(self.proj(input))
      proj_gate = nn.functional.sigmoid(self.transform(input))
      gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
      return gated

class HNetwork( nn.Module ):
   def __init__( self ):
      super().__init__()
      self.fc1 = nn.Linear( 512, 256 )
      self.highways = nn.ModuleList( [ Highway( 256 ) for _ in range( 5 ) ] )
      self.fc2 = nn.Linear( 256, 512 )
   
   def forward( self, x ):
      hidden = self.fc1( x )
      for h in self.highways:
         hidden = h( hidden )
      return self.fc2( hidden )
   
class HNetworkList( nn.Module ):
   def __init__( self ):
      super().__init__()
      self.networks = nn.ModuleList( [ HNetwork() for _ in range( n_latent ) ] )

   def forward( self, x ):
      outputs = []
      for i in range( n_latent ):
         out = self.networks[ i ]( x[ :, i ] )
         outputs.append( out.unsqueeze( 1 ) )
      return torch.cat( outputs, dim=1 )