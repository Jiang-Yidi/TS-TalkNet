# TS-TalkNet


> [**Target Active Speaker Detection with Audio-visual Cues**](https://arxiv.org/abs/2305.12831)<br>
> [Yidi Jiang](https://scholar.google.com/citations?user=le6gC58AAAAJ&hl=en&oi=ao), [Ruijie Tao](https://scholar.google.com/citations?user=sdXITx8AAAAJ&hl=en), [Zexu Pan](https://scholar.google.com/citations?user=GGIBU74AAAAJ&hl=en), [Haizhou Li](https://colips.org/~eleliha/)<br>
> NUS; CUHK <br>
> INTERSPEECH 2023

![image](https://github.com/Jiang-Yidi/TS-TalkNet/blob/main/overview.png)

The overview framework of our TS-TalkNet. It consists of a feature representation frontend and a speaker detection backend classifier. The feature representation frontend includes audio and visual temporal encoders, and speaker encoder. The speaker detection backend comprises a cross-attention and a fusion module to combine the audio, visual and speaker embeddings, and a self-attention module to predict the ASD scores. The lock represents the speaker encoder is frozen in our framework.

## TS-TalkNet in AVA-Activespeaker dataset

#### Data preparation

I follow the same prepocessing for AVA dataset as [TalkNet](https://arxiv.org/pdf/2107.06592.pdf). The details can be found in [here](https://github.com/TaoRuijie/TalkNet_ASD/blob/main/utils/tools.py#L34).

The following script can be used to download and prepare the AVA dataset for training.

```
python train.py --dataPathAVA AVADataPath --download 
```

`AVADataPath` is the folder you want to save the AVA dataset and its preprocessing outputs

#### Face-speaker library

You should run the data_prep/face_enroll_speech.py file to construct the face-speaker library and save to 'enrollmentPath'.

#### Training
Then you can train TalkNet in AVA end-to-end by using:
```
python train.py --dataPathAVA AVADataPath --enroll_speech_folder enrollmentPath
```
`exps/exps1/score.txt`: output score file, `exps/exp1/model/model_00xx.model`: trained model,


### Citation

Please cite the following if our paper or code is helpful to your research.
```
@inproceedings{jiang2023target,
  title={Target Active Speaker Detection with Audio-visual Cues},
  author={Jiang, Yidi and Tao, Ruijie and Pan, Zexu and Li, Haizhou},
  booktitle={Proc. Interspeech},
  year={2023}
}

@inproceedings{tao2021someone,
  title={Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection},
  author={Tao, Ruijie and Pan, Zexu and Das, Rohan Kumar and Qian, Xinyuan and Shou, Mike Zheng and Li, Haizhou},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  pages = {3927–3935},
  year={2021}
}
```


