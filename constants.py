e4e_weights_path = 'weights/e4e.pt'
stylegan2_weights_path = 'weights/stylegan2.pt'
styleExtractor_weights_path = 'weights/styleExtractor.pt'
style_extraction_original_path = 'weights/style_extraction_original.pth'
face_parsing_weights_path = 'weights/face_parsing.pth'
hopenet_weights_path = 'hopenet/model/shuff_epoch_120.pkl'
images_path = 'celeba_hq_256/'
latent_space_codes_path = 'latent_space_codes/'
cuda = False
mps = True
n_latent = 18
resolution = 256
device = 'cuda' if cuda else 'mps' if mps else 'cpu'
coeffs = [ 1, 3.5, 1, 0.1 ]
folders = {
   'train': 'data/train',
   'test': 'data/test'
}