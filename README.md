# TS-TalkNet


> [**Target Active Speaker Detection with Audio-visual Cues**](https://arxiv.org/abs/2305.12831)<br>
> [Yidi Jiang](https://scholar.google.com/citations?user=le6gC58AAAAJ&hl=en&oi=ao), [Ruijie Tao](https://scholar.google.com/citations?user=sdXITx8AAAAJ&hl=en), [Zexu Pan](https://scholar.google.com/citations?user=GGIBU74AAAAJ&hl=en), [Haizhou Li](https://colips.org/~eleliha/)<br>
> National University of Singapore, Singapore; The Chinese University of Hong Kong, Shenzhen, China
> INTERSPEECH 2023

![image](https://github.com/Jiang-Yidi/TS-TalkNet/blob/main/overview.png)

The overview framework of our TS-TalkNet. It consists of a feature representation frontend and a speaker detection backend classifier. The feature representation frontend includes audio and visual temporal encoders, and speaker encoder. The speaker detection backend comprises a cross-attention and a fusion module to combine the audio, visual and speaker embeddings, and a self-attention module to predict the ASD scores. The lock represents the speaker encoder is frozen in our framework.


