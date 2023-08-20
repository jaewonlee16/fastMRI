import torch
import torch.nn as nn

from varnet import VarNet
from utils.classification.binary_classifier import Unet_classifier

class integrated_net(nn.Module):
    def __init__(self, device,
                 class_dir,
                 acc4_model_dir,     # '../result/acc4_varnet/checkpoints/best_model.pt'
                 acc8_model_dir,
                 classifier_in_chans = 1, 
                 classifier_out_chans = 1, 
                 acc4_num_cascades = 4, 
                 acc4_chans = 18, 
                 acc4_sens_chans = 8, 
                 acc8_num_cascades = 4, 
                 acc8_chans = 18, 
                 acc8_sens_chans = 8
                ):
        super().__init__()
        
        # Initialization
        self.classifier = Unet_classifier(classifier_in_chans, classifier_out_chans).to(device)
        self.acc4_varnet = VarNet(num_cascades=acc4_num_cascades, 
                                  chans=acc4_chans, 
                                  sens_chans=acc4_sens_chans
                                 ).to(device)
        self.acc8_varnet = VarNet(num_cascades=acc8_num_cascades, 
                                  chans=acc8_chans, 
                                  sens_chans=acc8_sens_chans
                                 ).to(device)
        
        # Load trained models
        checkpoint = torch.load(class_dir, map_location='cpu')
        print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
        self.classifier.load_state_dict(checkpoint['model'])
        
        checkpoint = torch.load(acc4_model_dir, map_location='cpu')
        print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
        self.acc4_varnet.load_state_dict(checkpoint['model'])
        
        checkpoint = torch.load(acc8_model_dir, map_location='cpu')
        print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
        self.acc8_varnet.load_state_dict(checkpoint['model'])
        
    def forward(self, kspace, mask):
        is_acc4 = torch.argmax(self.classifier(kspace, mask))
        #print(f'{is_acc4=}')
        if is_acc4.cpu().item() == 1:
            return self.acc4_varnet(kspace, mask)
        else:
            return self.acc8_varnet(kspace, mask)