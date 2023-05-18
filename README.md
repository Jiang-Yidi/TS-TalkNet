# TS-TalkNet

# INTERSPEECH2023: Target Active Speaker Detection with Audio-visual Cues

![image](https://github.com/Jiang-Yidi/TS-TalkNet/overview.pdf)

The overview framework of our TS-TalkNet. It consists of a feature representation frontend and a speaker detection backend classifier. The feature representation frontend includes audio and visual temporal encoders, and speaker encoder. The speaker detection backend comprises a cross-attention and a fusion module to combine the audio, visual and speaker embeddings, and a self-attention module to predict the ASD scores. The lock represents the speaker encoder is frozen in our framework.


