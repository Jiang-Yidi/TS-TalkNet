import os, subprocess,cv2,random
import glob
import tqdm
import json 
from scipy.io import wavfile
import numpy as np
import csv
from face_recognition import *
from collections import defaultdict
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

orig_videoPath = './orig_videos/'
orig_audioPath = './orig_audios'
clip_audioFolder = './clips_audios/'
clip_videoFolder = './clips_videos/'
enrol_audioPath = './enroll_audios_0.7/'
csv_path = './csv/'


class face_recog(nn.Module):
	def __init__(self):
		super(face_recog, self).__init__()
		self.face_encoder    = IResNet(model ='res50').cuda()
		loadedState = torch.load('V-Glint.model', map_location="cuda")
		selfState = self.face_encoder.state_dict()
		for name, param in loadedState.items():
			origName = name
			if name not in selfState:
				continue
			if selfState[name].size() != loadedState[origName].size():
				sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
				continue
			selfState[name].copy_(param)
		for param in self.face_encoder.parameters():
			param.requires_grad = False

	def load_face(self,frame):
		frame = cv2.imread(frame)			
		face = cv2.resize(frame, (112, 112))
		face = np.transpose(face, (2, 0, 1))
		face = torch.FloatTensor(np.array(face))
		return face

	def get_embed(self,path):
		self.eval()
		frames = glob.glob("%s/*.jpg"%(path))
		frames = random.sample(frames,10)
		avg_embed = []
		for frame in frames:
			face = self.load_face(frame)
			face = face.div_(255).sub_(0.5).div_(0.5)
			face = face.unsqueeze(0)
			v_embedding = self.face_encoder.forward(face.cuda())
			v_embedding = v_embedding.detach().cpu().numpy()
			avg_embed.append(v_embedding)
		avg_embed = np.array(avg_embed).mean(0)
		return avg_embed


def face_similarity(x, y):
    return cosine_similarity(x, y)


def get_label(dataType,name):
	loader_csv = os.path.join(csv_path, '%s_loader.csv'%(dataType))
	lines = open(loader_csv).read().splitlines()
	for line in lines:
		data = line.split('\t')
		line_name = data[0]
		if line_name==name:
			res = []
			labels = data[3].replace('[', '').replace(']', '')
			labels = labels.split(',')
			for label in labels:
				res.append(int(label))
			res = np.array(res)
	return res


def face_speaker_process():
	for dataType in ['train', 'val']:
		enrol_audioPath = os.path.join('./enroll_audios_0.7/', '%s'%(dataType))
		os.makedirs(enrol_audioPath, exist_ok = True)

		videos = glob.glob("%s/*"%(os.path.join(clip_videoFolder, dataType)))
		for videoPath in tqdm.tqdm(videos):
			enrolaudio_folder = os.path.join(enrol_audioPath, videoPath.split('/')[-1])
			os.makedirs(enrolaudio_folder, exist_ok = True)
			clipaudio_folder = os.path.join(clip_audioFolder, dataType, videoPath.split('/')[-1])

			trackFolders = glob.glob("%s/*"%(videoPath))
			

			num_track = len(trackFolders)
			track_visual_embed = defaultdict(list)
			for trackFolder in trackFolders:
				#print(trackFolder)
				trackName = trackFolder.split('/')[-1]
				track_visual_embed[trackName] = face_recog().get_embed(trackFolder)

			for track1 in track_visual_embed.keys():
				os.makedirs(os.path.join(enrolaudio_folder, track1), exist_ok = True)
				for track2 in track_visual_embed.keys():
					similarity_score = face_similarity(track_visual_embed[track1],track_visual_embed[track2])[0]
					if similarity_score>0.7:
						sr, track2_audio = wavfile.read('%s/%s'%(clipaudio_folder, track2 + '.wav'))

						label_lst2 = get_label(dataType,track2)
						label_lst2 = label_lst2.repeat(640)
						track2_audio = track2_audio[:len(label_lst2)]

						if len(track2_audio) < len(label_lst2):
							shortage = len(label_lst2) - len(track2_audio)
							track2_audio = np.pad(track2_audio, (0, shortage), 'wrap') 

						enroll_speech =  track2_audio*label_lst2
						enroll_speech = enroll_speech[enroll_speech != 0]

						if len(enroll_speech)>16000:  

							enroll_speech = enroll_speech.astype(np.int16)
							enroll_speech_path = os.path.join(enrolaudio_folder, track1, track2+'.wav')
							wavfile.write(enroll_speech_path, sr, enroll_speech)		


if __name__ == '__main__':
    face_speaker_process()