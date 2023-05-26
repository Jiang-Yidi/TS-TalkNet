import os, numpy, cv2, random, glob, python_speech_features
import torch
from scipy.io import wavfile
import soundfile
from torchvision.transforms import RandomCrop



def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    cleanDB    = 10 * numpy.log10(max(1e-4, numpy.mean(audio ** 2)))
    noiseDB    = 10 * numpy.log10(max(1e-4, numpy.mean(noiseAudio ** 2)))
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)


def load_audio(data, dataPath, enroll_speech_folder, numFrames, label, audioAug, audioSet = None, p = True):
    dataName = data[0]
    fps = float(data[2])   
    audio = audioSet[dataName]
    label = label.repeat(640)

    enrol_folder = os.path.join(enroll_speech_folder,dataName[:11],dataName)
    audioFiles = glob.glob("%s/*.wav"%(enrol_folder))
    #print(len(audioFiles))
    if len(audioFiles)==0:
        ref_speech = numpy.zeros(len(label))
    else:
        audioFile = random.choice(audioFiles)
        _, ref_speech = wavfile.read(audioFile)
            
        if len(ref_speech) < len(label):
            shortage = len(label) - len(ref_speech)
            ref_speech = numpy.pad(ref_speech, (0, shortage), 'wrap') 

    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio, ref_speech[:len(label)]

def load_visual(data, dataPath, numFrames, visualAug): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces

def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, enroll_speech_folder, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.enroll_speech_folder = enroll_speech_folder
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)            
        start = 0        
        while True:
          length = int(sortedMixLst[start].split('\t')[1])
          end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedMixLst[start:end])
          if end == len(sortedMixLst):
              break
          start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, ref_speechs, labels = [], [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        
        for line in batchList:
            data = line.split('\t')   
            label = load_label(data, numFrames)
            audioFeature, ref_speech = load_audio2(data, self.audioPath, self.enroll_speech_folder, numFrames, label = label, audioAug = True, audioSet = audioSet, p = True)  
            audioFeatures.append(audioFeature)
            ref_speechs.append(ref_speech)
            visualFeatures.append(load_visual(data, self.visualPath, numFrames, visualAug = True))
            labels.append(label)
        audioFeatures = torch.FloatTensor(numpy.array(audioFeatures))
        visualFeatures = torch.FloatTensor(numpy.array(visualFeatures))
        ref_speechs = torch.FloatTensor(numpy.array(ref_speechs))
        labels = torch.LongTensor(numpy.array(labels))
        return audioFeatures, \
               visualFeatures, \
               ref_speechs, \
               labels, \

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.enroll_speech_folder = '/home/panzexu/yidi/AVA_dataset/enroll_audios_0.7/val/'
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        label = load_label(data, numFrames) 
        audioFeatures, ref_speechs = load_audio2(data, self.audioPath, self.enroll_speech_folder, numFrames, label = label, audioAug = False, audioSet = audioSet, p = False)
        audioFeatures = [audioFeatures]
        ref_speechs = [ref_speechs]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        labels = [label]
        audioFeatures = torch.FloatTensor(numpy.array(audioFeatures))
        visualFeatures = torch.FloatTensor(numpy.array(visualFeatures))
        ref_speechs = torch.FloatTensor(numpy.array(ref_speechs))
        labels = torch.LongTensor(numpy.array(labels))
        return audioFeatures, \
               visualFeatures, \
               ref_speechs, \
               labels, \

    def __len__(self):
        return len(self.miniBatch)
