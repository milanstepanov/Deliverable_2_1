# A simple wrapper around original implementation of view synthesis code
# which allows defining the network in MATLAB and inferring the provided
# input.

import network_v2 as networks
import torch
import numpy as np

class OAVS():
    
    def __init__(self, path_to_model, use_cuda=True):
        
        # TODO: check if the model at the path exists        
        self.use_cuda = use_cuda
        
        print("Init...")
        self.net = networks.OcclusionAwareVS(angular=7, dmax=4, use_cuda=use_cuda)
        
        try: # load model

            checkpoint = torch.load(path_to_model) # path_to_best_model
            print("Loaded checkpoint ", path_to_model)

            self.net.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded state")
            self.net.eval()
            
            print('Model loaded.')

        except:
            
            print('Model not loaded.')
        
        
    def forward(self, sample):
        
        with torch.no_grad():                
            prediction = self.net.forward(sample['p'], sample['q'],
                                          sample['c1'], sample['c2'],
                                          sample['c3'], sample['c4'])
            
            out = np.ascontiguousarray(prediction[0].permute((0,2,3,1)).cpu().detach().numpy(), dtype=np.float32)*255
            disp = np.ascontiguousarray(prediction[3].permute((0,2,3,1)).cpu().detach().numpy(), dtype=np.float32)
            m = np.ascontiguousarray(prediction[2].permute((0,2,3,1)).cpu().detach().numpy(), dtype=np.float32)
            
            return {"pred": out.astype(np.uint8), "disp": disp, "m": m}
        