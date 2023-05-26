import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
#from model.ts_talkNetModel import ts_talkNetModel
#from model.ablation1_talkNetModel import ts_talkNetModel
from model.ablation2_talkNetModel import ts_talkNetModel
from subprocess import PIPE
from torch.cuda.amp import autocast,GradScaler

class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(talkNet, self).__init__()     

        self.model = ts_talkNetModel().cuda()
        self.lossAV = lossAV(C = 128 * 2 + 192).cuda()
        self.lossA = lossA(C = 128).cuda()
        self.lossV = lossV(C = 128).cuda()

        ###fusion-ablation1
        # self.model = ts_talkNetModel().cuda()
        # self.lossAV = lossAV(C = 192 * 2).cuda()
        # self.lossA = lossA(C = 96).cuda()
        # self.lossV = lossV(C = 96).cuda()

        ###fusion-ablation2
        # self.model = ts_talkNetModel().cuda()
        # self.lossAV = lossAV(C = 192 * 3).cuda()
        # self.lossA = lossA(C = 192).cuda()
        # self.lossV = lossV(C = 192).cuda()

        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']    
        scaler = GradScaler()  
        time_start = time.time()    
        for num, (audioFeature, visualFeature, ref_speech, labels) in enumerate(loader, start=1):
            self.zero_grad()
            with autocast():
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda()) # feedForward
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                with torch.no_grad():   
                    speakerEmbed = self.model.forward_speaker_encoder(ref_speech[0].cuda()).detach()
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, speakerEmbed) 
                outsA = self.model.forward_audio_backend(audioEmbed)
                outsV = self.model.forward_visual_backend(visualEmbed)
                labels = labels[0].reshape((-1)).cuda() # Loss
                nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
                nlossA = self.lossA.forward(outsA, labels)
                nlossV = self.lossV.forward(outsV, labels)
                nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
                loss += nloss.detach().cpu().numpy()
                top1 += prec
            scaler.scale(nloss).backward()
            scaler.step(self.optim)
            scaler.update()
            index += len(labels)
            time_used = time.time() - time_start
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, (est %.1f mins), "    %(epoch, lr, 100 * (num / loader.__len__()), time_used * loader.__len__() / num / 60) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature,ref_speech, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                speakerEmbed = self.model.forward_speaker_encoder(ref_speech[0].cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, speakerEmbed)  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
