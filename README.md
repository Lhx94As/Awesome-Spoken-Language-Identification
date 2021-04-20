# Awesome-Spoken-Language-Identification
An awesome spoken LID repository. (Working in progress  
I made this repository is to help people who are working alone for LID and fresh to this area. So feel free to discuss with me by email, etc.  
If you know some interesting papers, challenge, etc. that I haven't put in this repo, feel free to tell me!  
# Useful Links:  
## Datasets:
[NIST Language Recognition Evaluation](https://www.nist.gov/itl/iad/mig/language-recognition) (The most famous and commonly used data)  
[openslr](https://openslr.org/resources.php) (open-source, plenty of datasets!)  
[Common-Voice](https://commonvoice.mozilla.org/en) (open-source, thousands of hours)  
## Workshops/Challenges:  
1. [OLR Challenge](http://cslt.riit.tsinghua.edu.cn/mediawiki/index.php/OLR_Challenge_2020): An Annual challenge, it is a special session in Interspeech 2021.  
2. [NIST LRE (the latest is LRE 2017)](https://www.nist.gov/itl/iad/mig/language-recognition): The NIST LRE data are the most commonly used data for LID task in Interspeech, ICASSP and other top conferences/journals.  
3. [1st WSTCSMC](https://www.microsoft.com/en-us/research/event/workshop-on-speech-technologies-for-code-switching-2020/): A workshop on Speech Technologies for Code-switching in Multilingual Communities.  
## Toolkits:
1. [Kaldi](https://kaldi-asr.org/) (Gorgeous toolkit, the LID parts are egs/LRE and egs/LRE07, developed by Dan Povey's team)  
2. [ASV-Subtools](https://github.com/Snowdar/asv-subtools#2-ap-olr-challenge-2020-baseline-recipe-language-identification) (A toolkit bases on kaldi&pytorch, developed by Xiamen University's team)  
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
#### x-vector: A common baseline NN-based approach, developed by JHU CLSP. (SOTA in some tasks, Embedding+back-end scoring)
Odessey 2018: [Spoken Language Recognition using X-vectors](https://www.danielpovey.com/files/2018_odyssey_xvector_lid.pdf)  
#### Other Deep Learning appoaches:
**Odessey 2018**:  
1.1 [Exploring the Encoding Layer and Loss Function in End-to-End Speaker and Language Recognition System](https://arxiv.org/pdf/1804.05160.pdf)  
1.2 [Spoken Language Recognition using X-vectors](https://www.danielpovey.com/files/2018_odyssey_xvector_lid.pdf)  
**Interspeech 2018**:  
2.1  
**ICASSP 2018**:  
3.1 [Insights into End-to-End Learning Scheme for Language Identification](https://arxiv.org/pdf/1804.00381.pdf)  
**Interspeech 2019**:  
4.1 [Survey Talk - End-to-End Deep Neural Network Based Speaker and Language Recognition](https://sites.duke.edu/dkusmiip/files/2019/09/IS19_Survey_SRELRE_MingLi_v2.pdf)  
4.2 [Contextual Phonetic Pretraining for End-to-end Utterance-level Language and Speaker Recognition](https://arxiv.org/pdf/1907.00457.pdf) (Test on LRE07)  
4.3 [A New Time-Frequency Attention Mechanism for TDNN and CNN-LSTM-TDNN, with Application to Language Identification](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1256.pdf) (Test on LRE07)  
4.4 [The XMUSPEECH System for the AP19-OLR Challenge](http://www.interspeech2020.org/uploadfile/pdf/Mon-1-11-2.pdf) (Test on AP19-OLR, best performer)  
4.5 [Attention Based Hybrid i-Vector BLSTM Model for Language Recognition](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2371.pdf) (Test on LRE17 and noisy data)
4.6 [Language Recognition using Triplet Neural Networks](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2437.pdf) (Test on LRE09, 15, 17)  
**ICASSP 2019**:  
5.1 [Interactive Learning of Teacher-student Model for Short Utterance Spoken Language Identification](https://ieeexplore.ieee.org/document/8683371)  
5.2 [Utterance-level End-to-End Language Identification using Attention based CNN-BLSTM](https://arxiv.org/pdf/1902.07374.pdf)(Test on LRE07)  
5.3 [End-to-end Language Recognition Using Attention Based Hierarchical Gated Recurrent Unit Models](https://ieeexplore.ieee.org/document/8683895) (Test on LRE2017)  
**Interspeech 2020**:  
6.1   
**ICASSP 2020**:  
7.1   
**Odeyssey 2020**:  
8.1 [BERTPHONE: Phonetically-aware Encoder Representations for Utterance-level Speaker and Language Recognition](https://www.isca-speech.org/archive/Odyssey_2020/pdfs/93.pdf)  
**TASLP**:  
9.1 2020 volume 28 [Towards Relevance and Sequence Modeling in Language Recognition](https://ieeexplore.ieee.org/document/9052484)
### Phonotactic:  
### Prosody:  
### Lexical:  
## Performance comparison:
![plot](https://github.com/Lhx94As/Awesome-Spoken-Language-Identification/blob/main/performance_.png)  
**Note that the results of GMM ivector, DNN ivector and xvector are reported in 4.3, and all evaluations are conducted on NIST LRE 07**  
