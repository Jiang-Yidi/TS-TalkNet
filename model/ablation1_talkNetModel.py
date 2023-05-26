import torch
import torch.nn as nn

from model.audioEncoder      import audioEncoder
from model.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from model.attentionLayer    import attentionLayer
from model.speakerEncoder    import ECAPA_TDNN

class ts_talkNetModel(nn.Module):
    def __init__(self):
        super(ts_talkNetModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D(out = 96)   # Visual Temporal Network Conv1d

        # Audio Temporal Encoder 
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 96])

        # Speaker Encoder
        self.speaker_encoder = ECAPA_TDNN(C = 1024)
        loadedState = torch.load('exps/pretrain.model', map_location="cuda")

        selfState = self.state_dict()
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False
        
        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = 96, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 96, nhead = 8)

        ##audio-speaker cross attention
        self.crossA2S = attentionLayer(d_model = 192, nhead = 8)
        self.crossS2A = attentionLayer(d_model = 192, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 192 * 2, nhead = 8)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)        
        x = self.audioEncoder(x)
        return x

    def forward_speaker_encoder(self, x):
        x = self.speaker_encoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1) 
        return x1_c, x2_c

    def forward_speaker_av_cross_attention(self, x1,x2):
        x1_c = self.crossA2S(src = x1, tar = x2)
        x2_c = self.crossS2A(src = x2, tar = x1)        
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2, x3): 
        x_av = torch.cat((x1,x2), 2)

        x3 = x3.unsqueeze(1)
        x3 = x3.repeat(1, x1.shape[1], 1)

        x_s_c, x_av_c = self.forward_speaker_av_cross_attention(x_av,x3)
        x_avs = torch.cat((x_s_c,x_av_c), 2)
        x_avs = self.selfAV(src = x_avs, tar = x_avs)       
        x_avs = torch.reshape(x_avs, (-1, 192*2))
        return x_avs   


    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 96))
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 96))
        return x

