import os
import shutil
import random
from constants import folders

celeba_hq_256_path = 'celeba_hq_256/'
fns = [ fn for fn in os.listdir( celeba_hq_256_path ) if fn.split( '.' )[ -1 ] in [ 'png', 'jpg', 'jpeg' ] ]

train = random.sample( fns, 27000 )
test = [ fn for fn in fns if fn not in train ]

os.makedirs( 'latent_space_codes', exist_ok=True )

os.makedirs( os.path.join( folders[ 'train' ], 'images' ), exist_ok=True )
os.makedirs( os.path.join( folders[ 'test' ], 'images' ), exist_ok=True )

for fn in train:
   shutil.copy( os.path.join( celeba_hq_256_path, fn ), os.path.join( folders[ 'train' ], 'images', fn ) )

for fn in test:
   shutil.copy( os.path.join( celeba_hq_256_path, fn ), os.path.join( folders[ 'test' ], 'images', fn ) )