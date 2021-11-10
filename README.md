# Awesome-Spoken-Language-Identification
An awesome spoken LID repository. (Working in progress  
I made this repository to help people who are working alone for LID and fresh to this area. So feel free to discuss with me by email, etc.  
If you know some interesting papers, challenge, etc. which I haven't put in this repo, feel free to tell me!  
  
I also included my pytorch re-implementation in folder models.
Updated descriptions for some papers. For those I didn't, either I forgot what it is exactly about or the title is too straightforward.  
# News:  
1. The OLR 2021 is opening for participanting! It's very worth trying! (10th Aug 2021)  
 
# Useful Links:  
## Datasets:
[NIST Language Recognition Evaluation](https://www.nist.gov/itl/iad/mig/language-recognition) (The most famous and commonly used data)  
[openslr](https://openslr.org/resources.php) (open-source, plenty of datasets!)  
[Common-Voice](https://commonvoice.mozilla.org/en) (open-source, thousands of hours)  
[OLR dataset](http://cslt.riit.tsinghua.edu.cn/mediawiki/index.php/OLR_Challenge_2020) Available to participants of the challenge.  
## Workshops/Challenges:  
1. [OLR Challenge](http://cslt.riit.tsinghua.edu.cn/mediawiki/index.php/OLR_Challenge_2020): An Annual challenge, it is a special session in Interspeech 2021.  
2. [NIST LRE (the latest is LRE 2017)](https://www.nist.gov/itl/iad/mig/language-recognition): The NIST LRE data are the most commonly used data for LID task in Interspeech, ICASSP and other top conferences/journals.  
3. [1st WSTCSMC](https://www.microsoft.com/en-us/research/event/workshop-on-speech-technologies-for-code-switching-2020/): A workshop on Speech Technologies for Code-switching in Multilingual Communities.  
## Toolkits:
1. [Kaldi](https://kaldi-asr.org/) (Gorgeous toolkit, the LID parts are egs/LRE and egs/LRE07, developed by Dan Povey's team)  
2. [ASV-Subtools](https://github.com/Snowdar/asv-subtools#2-ap-olr-challenge-2020-baseline-recipe-language-identification) (A toolkit bases on **kaldi&pytorch**, developed by Xiamen University's team)  
3. [SpeechBrain](https://speechbrain.github.io/index.html)(SpeechBrain is an open-source and all-in-one speech toolkit relying on **PyTorch**, yet I have never tried it so far)  
4. [lidbox](https://github.com/py-lidbox/lidbox) (Some implementations of models in recent Interspeech conferences on **tensorflow**)  
5. [WeNet](https://github.com/wenet-e2e)  
6. [ESPNet](https://github.com/espnet)
7. [s3prl](https://github.com/Lhx94As/s3prl)  
## Papers:  
Apart from these LID papers, other papers about ASR, Speaker Verification/Recognition and Speaker Diarization are also worth reading.  
(from 2018 to now, I am always updating the paper list)  
An overview (strongly recommend): [Spoken Language Recognition: from fundamental to practice](https://ieeexplore.ieee.org/document/6451097)  
### Acoustic Phonetic:  
#### i-vector: A common baseline GMM-UBM-based approach. Out of date but worth reading.  
TASLP 2005: [Eigenvoice Modeling With Sparse Training Data](https://www.crim.ca/perso/patrick.kenny/eigenvoices.PDF)  
TASLP 2007: [A Joint factor analysis approach to progressive model adaptation in text independent speaker verification](https://www.crim.ca/perso/patrick.kenny/IEEETrans07_Yin.pdf)  
Interspeech 2011: [Language Recognition via Ivectors and Dimensionality Reduction](https://groups.csail.mit.edu/sls/publications/2011/Dehak_Interspeech11.pdf)  
#### x-vector: A common baseline NN-based approach, developed by JHU CLSP. (SOTA in some tasks, Embedding+back-end scoring)
Odyssey 2018: [Spoken Language Recognition using X-vectors](https://www.danielpovey.com/files/2018_odyssey_xvector_lid.pdf)  

#### Other Deep Learning appoaches:
**Odyssey 2018**:  
1.1 [Exploring the Encoding Layer and Loss Function in End-to-End Speaker and Language Recognition System](https://arxiv.org/pdf/1804.05160.pdf)  
>Kind of summary, there is a LDE layer which is interesting.  

1.2 [Spoken Language Recognition using X-vectors](https://www.danielpovey.com/files/2018_odyssey_xvector_lid.pdf)  
>JUST READ IT! The most commonly used baseline for now!  

**Interspeech 2018**:  
2.1 [Sub-band Envelope Features Using Frequency Domain Linear Prediction for Short Duration Language Identification](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1805.pdf)  
**ICASSP 2018**:  
3.1 [Insights into End-to-End Learning Scheme for Language Identification](https://arxiv.org/pdf/1804.00381.pdf)  
**Interspeech 2019**:  
4.1 [Survey Talk - End-to-End Deep Neural Network Based Speaker and Language Recognition](https://sites.duke.edu/dkusmiip/files/2019/09/IS19_Survey_SRELRE_MingLi_v2.pdf)  
4.2 [Contextual Phonetic Pretraining for End-to-end Utterance-level Language and Speaker Recognition](https://arxiv.org/pdf/1907.00457.pdf) (Test on LRE07)  
4.3 [A New Time-Frequency Attention Mechanism for TDNN and CNN-LSTM-TDNN, with Application to Language Identification](https://www.isca-speech.org/archive/pdfs/interspeech_2019/miao19b_interspeech.pdf) (Test on LRE07)
>Attention mechanism in time domain and frequency domain and integrate them into one model (CNN-LSTM-TDNN)  

4.4 [The XMUSPEECH System for the AP19-OLR Challenge](http://www.interspeech2020.org/uploadfile/pdf/Mon-1-11-2.pdf) (Test on AP19-OLR, best performer)  
4.5 [Attention Based Hybrid i-Vector BLSTM Model for Language Recognition](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2371.pdf) (Test on LRE17 and noisy data)  
>Two stage processing, first i-vector extraction in 200ms level with 100ms overlap. Then feed the i-vectors into a BLSTM model for LID.

4.6 [Language Recognition using Triplet Neural Networks](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2437.pdf) (Test on LRE09, 15, 17)  
**ICASSP 2019**:  
5.1 [Interactive Learning of Teacher-student Model for Short Utterance Spoken Language Identification](https://ieeexplore.ieee.org/document/8683371)  
>KD-based approach, this author proposed a series of papers for short-utterance LID not only based on KD but also other algorithms, see 8.2.  

5.2 [Utterance-level End-to-End Language Identification using Attention based CNN-BLSTM](https://arxiv.org/pdf/1902.07374.pdf)(Test on LRE07)  
>Apply self-attention for pooling  

5.3 [End-to-end Language Recognition Using Attention Based Hierarchical Gated Recurrent Unit Models](https://ieeexplore.ieee.org/document/8683895) (Test on LRE2017)  
> Stack two GRUs layers then self-attention for pooling (like 5.2).  

**Interspeech 2020 [Language Recognition Session](https://isca-speech.org/archive/Interspeech_2020/)**:  
6.1 [Metric Learning Loss Functions to Reduce Domain Mismatch in the x-Vector Space for Language Recognition](https://isca-speech.org/archive/Interspeech_2020/pdfs/1708.pdf)   
**ICASSP 2020**:  
7.1   
**Odyssey 2020**:  
8.1 [BERTPHONE: Phonetically-aware Encoder Representations for Utterance-level Speaker and Language Recognition](https://www.isca-speech.org/archive/pdfs/odyssey_2020/ling20_odyssey.pdf)  
>The same work as 4.2, just read one of them.  

8.2 [Compensation on x-vector for Short Utterance Spoken Language Identification](https://www.isca-speech.org/archive/Odyssey_2020/pdfs/66.pdf)  
**Interspeech 2021**:  
9.1 [Modeling and Training Strategies for Language Recognition Systems](https://www.isca-speech.org/archive/pdfs/interspeech_2021/duroselle21_interspeech.pdf)  
>This paper summarized how pre-trained ASR model influences the LID performance, worth reading if you are interested in such topic (wav2vec, etc.)  

9.2 [Self-Supervised Phonotactic Representations for Language Identification](https://www.isca-speech.org/archive/pdfs/interspeech_2021/ramesh21_interspeech.pdf)  
9.3 [A Weight Moving Average Based Alternate Decoupled Learning Algorithm for Long-Tailed Language Identification](https://www.isca-speech.org/archive/pdfs/interspeech_2021/wang21o_interspeech.pdf)  
9.4 [Serialized Multi-Layer Multi-Head Attention for Neural Speaker Embedding](https://www.isca-speech.org/archive/pdfs/interspeech_2021/zhu21c_interspeech.pdf)  
>Apply transformer encoder layers as a pooling module for speaker verification, but I think it can also work for LID  

**ICASSP 2021**:  
10.1  

**TASLP**:  
1. 2020 volume 28 [Towards Relevance and Sequence Modeling in Language Recognition](https://ieeexplore.ieee.org/document/9052484)  
>A summary of 4.5 and 5.3. (But the results of their baselines seems strange, the i-vector and x-vector may be exchanged)  

### Phonotactic:  
### Prosody:  
### Lexical:  
## Performance comparison:
![plot](https://github.com/Lhx94As/Awesome-Spoken-Language-Identification/blob/main/performance_.png)  
**Note that the results of GMM ivector, DNN ivector and xvector are reported in 4.3, and all evaluations are conducted on NIST LRE 07. Some paper used data augmentation which can improve the performance**  
