## ARCturus (v0.1)

The ARCturus (**A**ffect **R**ecognition and **C**ontrol) system is my attempt at building an emotion-recognition model that generates music in order to selectivly control negative emotions. A user's speech sample is first collected using a speech-recognition module (OpenAI's Whisper model, in this case) and is then fed into BERT in order to get a 3-dimensional VAD (valence-arousal-dominance) embedding of the user's emotion

The VAD embedding is fed into a generative model (ChatMusician), which generates a musical piece based on the values in the embedding.

