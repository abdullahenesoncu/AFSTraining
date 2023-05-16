import cv2
from PIL import Image
from AFS.models.afs import AFS
import torch
from constants import e4e_weights_path, style_extraction_original_path, stylegan2_weights_path, face_parsing_weights_path, device

afs = None
def afsInference( srcImagePath, tgtImagePath, resImagePath ):
    global afs
    if not afs:
        afs = AFS(ckpt_e4e=e4e_weights_path,
                ckpt_style_extraction=style_extraction_original_path,
                ckpt_face_parsing=face_parsing_weights_path,
                ckpt_stylegan=stylegan2_weights_path, device=device).to( device ).eval()
        print( "AFS CREATED" )
    src_img = cv2.imread(srcImagePath)
    src_img = cv2.resize( src_img, (1024, 1024) )
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img = ((torch.tensor(src_img) - 127.5) / 127.5).permute(2, 0, 1).unsqueeze(0).to(device)

    tgt_img = cv2.imread(tgtImagePath)
    tgt_img = cv2.resize( tgt_img, (1024, 1024) )
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
    tgt_img = ((torch.tensor(tgt_img) - 127.5) / 127.5).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out_img = afs(src_img, tgt_img)[0] * 127.5 + 127.5

    out_img = out_img.permute(1, 2, 0).cpu().detach().numpy()
    cv2.imwrite(resImagePath, out_img[:, :, ::-1])
