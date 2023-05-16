from PIL import Image
from hNetwork import HNetworkList
import torch
import torch.optim as optim
from data import getBatch, fns
from loss import computeLoss
from constants import styleExtractor_weights_path, device
from torch.utils.tensorboard import SummaryWriter

styleExtractor = HNetworkList().to( device )

try:
   checkpoint = torch.load( styleExtractor_weights_path )
   styleExtractor.load_state_dict( checkpoint )
   print( 'Model is loaded' )
except Exception as e:
   pass

start = 0
try:
   start = int( open( 'runs/iteration.txt', 'r' ).read() ) + 1
   print( f'Starting from epoch {start}' )
except:
   pass

lr = 1e-4
batchSize = 1
numEpoches = len( fns[ 'train' ] ) * 4 // batchSize

optimizer = optim.Adam( styleExtractor.parameters(), lr=lr )
lrScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpoches, eta_min=1e-6 )
writer = SummaryWriter()

for epoch in range( start, numEpoches ):
   optimizer.zero_grad()
   data = getBatch( batchSize )
   ws = torch.cat( [ x[ 0 ][ 1 ].unsqueeze( 0 ) for x in data ] )
   wt = torch.cat( [ x[ 1 ][ 1 ].unsqueeze( 0 ) for x in data ] )
   srcImages = torch.cat( [ x[ 0 ][ 0 ].unsqueeze( 0 ) for x in data ] )
   tarImages = torch.cat( [ x[ 1 ][ 0 ].unsqueeze( 0 ) for x in data ] )
   
   w_sty = styleExtractor( torch.cat( ( ws, wt ) ) )
   ws_sty = w_sty[ : ws.shape[ 0 ] ]
   wt_sty = w_sty[ ws.shape[ 0 ] : ]
   
   w_sty = styleExtractor( wt_sty + ws - ws_sty )
   
   loss = computeLoss( wt_sty + ws - ws_sty, wt, wt_sty, w_sty, srcImages, tarImages, writer, epoch )
   loss.backward()
   optimizer.step()
   lrScheduler.step()
   
   print( epoch, loss.item() )
   writer.add_scalar("Loss/train", loss, epoch)
   open( 'runs/iteration.txt', 'w' ).write( str( epoch ) )
   
   if epoch % 10 == 9:
      torch.save( styleExtractor.to('cpu').state_dict(), styleExtractor_weights_path )
      styleExtractor.to(device)
      
writer.flush()

