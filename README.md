# Awesome-Spoken-Language-Identification
An awesome spoken LID repository. (Working in progress  
I made this repository is to help people who are working alone for LID and fresh to this area. So feel free to discuss with me by email, etc.  
If you know some interesting papers, challenge, etc. that I haven't put in this repo, feel free to tell me!  
# Useful Links:  
## Datasets:  
[openslr](https://openslr.org/resources.php) (open-source, plenty of datasets!)  
[Common-Voice](https://commonvoice.mozilla.org/en) (open-source, thousands of hours)  
## Workshops/Challenges:  
1. [OLR Challenge](http://cslt.riit.tsinghua.edu.cn/mediawiki/index.php/OLR_Challenge_2020): An Annual challenge, it is a special session in Interspeech 2021.  
2. [NIST LRE (the latest is LRE 2017)](https://www.nist.gov/itl/iad/mig/language-recognition): The NIST LRE data are the most commonly used data for LID task in Interspeech, ICASSP and other top conferences/journals.  
3. [1st WSTCSMC](https://www.microsoft.com/en-us/research/event/workshop-on-speech-technologies-for-code-switching-2020/): A workshop on Speech Technologies for Code-switching in Multilingual Communities.  
## Toolkits:
1. [Kaldi](https://kaldi-asr.org/) (Gorgeous toolkit, the LID parts are egs/LRE and egs/LRE07)  
2. [XMUspeech](https://github.com/Snowdar/asv-subtools#2-ap-olr-challenge-2020-baseline-recipe-language-identification) (A toolkit bases on kaldi&pytorch, developed by Xiamen University's team)  
3. [SpeechBrain](https://speechbrain.github.io/index.html)(SpeechBrain is an open-source and all-in-one speech toolkit relying on PyTorch, yet I have never tried it so far)  
## Papers:  
Apart from these LID papers, other papers about ASR, Speaker Verification/Recognition and Speaker Diarization are also worth reading.  
(from 2018 to now, yet I am updating the paper list)  
An overview (strongly recommend): [Spoken Language Recognition: from fundamental to practice](https://ieeexplore.ieee.org/document/6451097)  
### Acoustic Phonetic:  
#### i-vector: A common baseline GMM-UBM-based approach. Out of date but worth reading.  
TASLP 2005: [Eigenvoice Modeling With Sparse Training Data](https://www.crim.ca/perso/patrick.kenny/eigenvoices.PDF)  
TASLP 2007: [A Joint factor analysis approach to progressive model adaptation in text independent speaker verification](https://www.crim.ca/perso/patrick.kenny/IEEETrans07_Yin.pdf)  
Interspeech 2011: [Language Recognition via Ivectors and Dimensionality Reduction](https://groups.csail.mit.edu/sls/publications/2011/Dehak_Interspeech11.pdf)  
#### x-vector: A common baseline NN-based Napproach, developed by Dan's team. (SOTA in some tasks, Embedding+back-end scoring)
Odessey 2018: [Spoken Language Recognition using X-vectors](https://www.danielpovey.com/files/2018_odyssey_xvector_lid.pdf)  
#### Other Deep Learning appoaches:
Interspeech 2018:  
ICASSP 2018:  
Interspeech 2019:  
[Contextual Phonetic Pretraining for End-to-end Utterance-level Language and Speaker Recognition](https://arxiv.org/pdf/1907.00457.pdf) (Test on LRE07)  
[The XMUSPEECH System for the AP19-OLR Challenge](http://www.interspeech2020.org/uploadfile/pdf/Mon-1-11-2.pdf) (Test on AP19-OLR, best performer)  
[A New Time-Frequency Attention Mechanism for TDNN and CNN-LSTM-TDNN, with Application to Language Identification](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1256.pdf)  
ICASSP 2019:  
[Utterance-level End-to-End Language Identification using Attention based CNN-BLSTM](https://arxiv.org/pdf/1902.07374.pdf)  
Interspeech 2020:  
ICASSP 2020:  
Odeyssey 2020:  
[BERTPHONE: Phonetically-aware Encoder Representations for Utterance-level Speaker and Language Recognition](https://www.isca-speech.org/archive/Odyssey_2020/pdfs/93.pdf)
### Phonotactic:  
### Prosody:  
### Lexical:  
