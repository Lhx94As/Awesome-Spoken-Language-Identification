# Awesome-Spoken-Language-Identification
A spoken LID repository, mainly focuses on conference papers since they are updated quickly.   
I made this repository to help people who are working alone for LID or fresh to this area. So feel free to discuss with me by email, etc.  

**Updated descriptions for some papers. For those I didn't, either I forgot what it is exactly about or the title is too straightforward.**  
**A PyTorch x-vector implementation was put in this repo, yet I recommend the asv-subtools, espnet, speechbrain, and kaldi recipe; they are better**
# News:  
1. I will update papers in ICASSP 2024 soon


# Recommended conferences&Journals:
Interspeech, Odyssey, ICASSP, APSIPA, ASRU, SLT, TSP, TASLP, JSTSP, Speech Comm, Signal Processing Letter, etc.  
# Useful Links:  
## Datasets:
[NIST Language Recognition Evaluation](https://www.nist.gov/itl/iad/mig/language-recognition) (The most famous and commonly used data)  
[openslr](https://openslr.org/resources.php) (open-source, plenty of datasets!)  
[Common-Voice](https://commonvoice.mozilla.org/en) (open-source, thousands of hours)  
[OLR dataset](http://cslt.riit.tsinghua.edu.cn/mediawiki/index.php/OLR_Challenge_2020) Available to participants of the challenge.  
[Voxlingual107](https://bark.phon.ioc.ee/voxlingua107/) I saw that more and more papers are using this dataset in Interspeech 2023 and NIST LRE 2022.  
[FLEURS](https://huggingface.co/datasets/google/fleurs) New trend for LID  
## Workshops/Challenges:  
1. [OLR Challenge](http://cslt.riit.tsinghua.edu.cn/mediawiki/index.php/OLR_Challenge_2020): An Annual challenge, it is a special session in Interspeech 2021.  
2. [NIST LRE (the latest is LRE 2022)](https://www.nist.gov/itl/iad/mig/language-recognition): The NIST LRE data are the most commonly used data for LID task in Interspeech, ICASSP and other top conferences/journals.  
3. [1st WSTCSMC](https://www.microsoft.com/en-us/research/event/workshop-on-speech-technologies-for-code-switching-2020/): A workshop on Speech Technologies for Code-switching in Multilingual Communities.  
## Toolkits:
1. [Kaldi](https://kaldi-asr.org/) (Gorgeous toolkit, the LID parts are egs/LRE and egs/LRE07, developed by Dan Povey's team)  
2. [hyperion](https://github.com/hyperion-ml/hyperion) (A speaker and language identification toolkit based on python, numpy, and torch)
3. [ASV-Subtools](https://github.com/Snowdar/asv-subtools#2-ap-olr-challenge-2020-baseline-recipe-language-identification) (A toolkit bases on **kaldi&pytorch**, developed by Xiamen University's team)  
4. [SpeechBrain](https://speechbrain.github.io/index.html)(SpeechBrain is an open-source and all-in-one speech toolkit relying on **PyTorch**, yet I have never tried it so far)  
5. [lidbox](https://github.com/py-lidbox/lidbox) (Some implementations of models in recent Interspeech conferences on **tensorflow**)  
6. [WeNet](https://github.com/wenet-e2e)  
7. [ESPNet](https://github.com/espnet) For E2E ASR, easy to learn, custom, and use.  
8. [s3prl](https://github.com/Lhx94As/s3prl) For wav2vec models application  
9. [hugging face](https://huggingface.co/docs/transformers/model_doc/wav2vec2) For wav2vec models  
## Papers:  
Apart from these LID papers, other papers about ASR, Speaker Verification/Recognition and Speaker Diarization are also worth reading.  
(from 2018 to now, I am always updating the paper list)  
An overview (strongly recommend): [Spoken Language Recognition: from fundamental to practice](https://ieeexplore.ieee.org/document/6451097)  
### i-vector: A common baseline GMM-UBM-based approach. Out of date but worth reading.  
TASLP 2005: [Eigenvoice Modeling With Sparse Training Data](https://www.crim.ca/perso/patrick.kenny/eigenvoices.PDF)  
TASLP 2007: [A Joint factor analysis approach to progressive model adaptation in text independent speaker verification](https://www.crim.ca/perso/patrick.kenny/IEEETrans07_Yin.pdf)  
Interspeech 2011: [Language Recognition via Ivectors and Dimensionality Reduction](https://groups.csail.mit.edu/sls/publications/2011/Dehak_Interspeech11.pdf)  
### x-vector: A common baseline NN-based approach, developed by JHU CLSP. (SOTA in some tasks, Embedding+back-end scoring)
Odyssey 2018: [Spoken Language Recognition using X-vectors](https://www.danielpovey.com/files/2018_odyssey_xvector_lid.pdf)  

### Other Deep Learning appoaches:
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
>Attention mechanism in time domain and frequency domain and integrate them into one model (CNN-LSTM-TDNN).  

4.4 [The XMUSPEECH System for the AP19-OLR Challenge](http://www.interspeech2020.org/uploadfile/pdf/Mon-1-11-2.pdf) (Test on AP19-OLR, best performer)  
4.5 [Attention Based Hybrid i-Vector BLSTM Model for Language Recognition](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2371.pdf) (Test on LRE17 and noisy data)  
>Two stage processing, first i-vector extraction in 200ms level with 100ms overlap. Then feed the i-vectors into a BLSTM model for LID.  

4.6 [Language Recognition using Triplet Neural Networks](https://www.isca-speech.org/archive/pdfs/interspeech_2019/mingote19b_interspeech.pdf) (Test on LRE09, 15, 17)  
 
**ICASSP 2019**:  
5.1 [Interactive Learning of Teacher-student Model for Short Utterance Spoken Language Identification](https://ieeexplore.ieee.org/document/8683371)  
>KD-based approach, this author proposed a series of papers for short-utterance LID not only based on KD but also other algorithms, see 8.2.  

5.2 [Utterance-level End-to-End Language Identification using Attention based CNN-BLSTM](https://arxiv.org/pdf/1902.07374.pdf)(Test on LRE07)  
>Apply self-attention for pooling with a CNN-BLSTM backbone.  

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

**ICASSP 2021**:  
9.1   

**Interspeech 2021**:  
10.1 [Modeling and Training Strategies for Language Recognition Systems](https://www.isca-speech.org/archive/pdfs/interspeech_2021/duroselle21_interspeech.pdf)  
>This paper summarized how pre-trained ASR model influences the LID performance, worth reading if you are interested in such topic (wav2vec, etc.)  

10.2 [Self-Supervised Phonotactic Representations for Language Identification](https://www.isca-speech.org/archive/pdfs/interspeech_2021/ramesh21_interspeech.pdf)  
>This paper mainly showed how effective the wav2vec features can be in a LID task, and tried to prove the existence of phonotactic information in wav2vec features.  

10.3 [A Weight Moving Average Based Alternate Decoupled Learning Algorithm for Long-Tailed Language Identification](https://www.isca-speech.org/archive/pdfs/interspeech_2021/wang21o_interspeech.pdf)  
>This paper introduced some training strategies to tackle the long-tailed LID task.  

10.4 [Serialized Multi-Layer Multi-Head Attention for Neural Speaker Embedding](https://www.isca-speech.org/archive/pdfs/interspeech_2021/zhu21c_interspeech.pdf)  
>Apply transformer encoder layers as a pooling module for speaker verification, but I think it can also work for LID.  

10.5 [End-to-End Language Diarization for Bilingual Code-Switching Speech](https://www.isca-speech.org/archive/pdfs/interspeech_2021/liu21d_interspeech.pdf)  
>TDNN layers + transformer encoder layers for bilingual code-switching LID (language diarization in this case), the last two TDNN layers are replaced by transformer encoder layers. (Code is available in my another repo)   

**ICASSP 2022**:  
11.1 [Phonotactic Language Recognition Using A Universal Phoneme Recognizer and A Transformer Architecture](https://ieeexplore.ieee.org/document/9746459)  
>Phonotactic LID with transformer encoder.  

11.2 [Spoken Language Recognition with Cluster-Based Modeling](https://ieeexplore.ieee.org/document/9747515)(test on LRE15 & OLR20)  

>Looks into cluster-based classifier for LID.  

11.3 [Improved Language Identification Through Cross-Lingual Self-Supervised Learning](https://arxiv.org/pdf/2107.04082.pdf)  

>Meta AI's work which extends wav2vec to LID, showing that XLSR-53 (multilingual wav2vec 2.0) is better than monolingual one and is able to achieve high accuracy with even very limited data for finetuning. This work also discussed the aggregation strategies (i.e., pooling layer), this part shows that mean + max + min is the best, which is a bit different from our experience that mean + std. pooling is better.  

**Odyssey 2022**:  
12.1 [Enhancing Language Identification using Dual-mode Model with Knowledge Distillation](https://www.isca-speech.org/archive/pdfs/odyssey_2022/liu22c_odyssey.pdf)(test on NIST LRE 2017)   
>Dual-mode LID, one for general speech, the other for its short clips. Show high performance improvement on both long and short-utterance LID. (My work, source code is in another repo of mine)  

12.2 [Attentive Temporal Pooling for Conformer-Based Streaming Language Identification in Long-Form Speech](https://www.isca-speech.org/archive/pdfs/odyssey_2022/wang22b_odyssey.pdf) 
>Streaming LID is achieved by accumulating the statistics frame by frame. 

12.3 [Pretraining Approaches for Spoken Language Recognition: TalTech Submission to the OLR 2021 Challenge](https://www.isca-speech.org/archive/pdfs/odyssey_2022/alumae22_odyssey.pdf)  
>Discuss many pre-training strategies for LID.  

**Interspeech 2022**:  
13.1 [PHO-LID: A Unified Model Incorporating Acoustic-Phonetic and Phonotactic Information for Language Identification](https://www.isca-speech.org/archive/interspeech_2022/liu22e_interspeech.html)(test on LRE17 & OLR17)   
>Incorporating phonotactic information in general LID model via a unsupervised phonem segmentation auxilary task. Show high performance improvement on long speech (advantage of phonotactic LID). (My work, source code available)  

13.2 [Oriental Language Recognition (OLR) 2021: Summary and Analysis](https://www.isca-speech.org/archive/interspeech_2022/wang22ga_interspeech.html)  

13.3 [Ant Multilingual Recognition System for OLR 2021 Challenge](https://www.isca-speech.org/archive/interspeech_2022/lyu22_interspeech.html)  

13.4 [Transducer-based language embedding for spoken language identification](https://www.isca-speech.org/archive/interspeech_2022/shen22b_interspeech.html)(test on Librispeech and VoxLingual)  
>Adopting the RNN-T framework to generate input features for LID model (stat. pooling + linears, in this paper). In other words, this paper incorporate text information and acoustic information. Kind of multi-modal from my understanding.  

**Interspeech 2023**:  
14.1 [Lightweight and Efficient Spoken Language Identification of Long-form Audio](https://www.isca-speech.org/archive/interspeech_2023/zhu23c_interspeech.html)  
14.2 [End-to-End Spoken Language Diarization with Wav2vec Embeddings](https://www.isca-speech.org/archive/interspeech_2023/mishra23_interspeech.html)  
14.3 [Efficient Spoken Language Recognition via Multilabel Classification](https://www.isca-speech.org/archive/interspeech_2023/nieto23_interspeech.html)  
14.4 [Description and Analysis of ABC Submission to NIST LRE 2022](https://www.isca-speech.org/archive/interspeech_2023/matejka23_interspeech.html)  
14.5 [Exploring the Impact of Pretrained Models and Web-Scraped Data for the 2022 NIST Language Recognition Evaluation](https://www.isca-speech.org/archive/interspeech_2023/alumae23_interspeech.html)   
14.6 [Advances in Language Recognition in Low Resource African Languages: The JHU-MIT Submission for NIST LRE22](https://www.isca-speech.org/archive/interspeech_2023/villalba23_interspeech.html)  
14.7 [What Can an Accent Identifier Learn? Probing Phonetic and Prosodic Information in a Wav2vec2-based Accent Identification Model](https://www.isca-speech.org/archive/interspeech_2023/yang23v_interspeech.html)  
14.8 [Description and analysis of the KPT system for NIST Language Recognition Evaluation 2022](https://www.isca-speech.org/archive/interspeech_2023/sarni23_interspeech.html)  
14.9 [Wavelet Scattering Transform for Improving Generalization in Low-Resourced Spoken Language Identification](https://www.isca-speech.org/archive/interspeech_2023/dey23_interspeech.html)  
14.10 [A Parameter-Efficient Learning Approach to Arabic Dialect Identification with Pre-Trained General-Purpose Speech Model](https://www.isca-speech.org/archive/interspeech_2023/radhakrishnan23_interspeech.html)  
14.11 [Self-supervised Learning Representation based Accent Recognition with Persistent Accent Memory](https://www.isca-speech.org/archive/interspeech_2023/li23aa_interspeech.html)  
14.12 [Multi-resolution Approach to Identification of Spoken Languages and To Improve Overall Language Diarization System Using Whisper Model](https://www.isca-speech.org/archive/interspeech_2023/vachhani23_interspeech.html)  
14.13 [MERLIon CCS Challenge: Multilingual Everyday Recordings - Language Identification On Code-Switched Child-Directed Speech](https://www.isca-speech.org/archive/interspeech_2023/index.html)  
>A language identification and diarization challenge on Singaporean English-Mandarin child-directed code-switching speech, is still open for new submissions.

14.14 [Conformer-based Language Embedding with Self-Knowledge Distillation for Spoken Language Identification](https://www.isca-speech.org/archive/interspeech_2023/wang23ia_interspeech.html)  
>Similar to 12.1.  

14.15 [A Compact End-to-End Model with Local and Global Context for Spoken Language Identification](https://www.isca-speech.org/archive/interspeech_2023/jia23b_interspeech.html)  
>Tranferred from an ASR model, the weighted cross-entropy is important in this implementation from my understanding.  

14.16 [Label Aware Speech Representation Learning For Language Identification](https://www.isca-speech.org/archive/interspeech_2023/vashishth23_interspeech.html)  
**TASLP**:  
1. 2020 volume 28 [Towards Relevance and Sequence Modeling in Language Recognition](https://ieeexplore.ieee.org/document/9052484)  
>A summary of 4.5 and 5.3. (But the results of their baselines seems strange, from my personal understanding, the i-vector and x-vector may be exchanged)  
2. 2022 volume 30 [A Discriminative Hierarchical PLDA-Based Model for Spoken Language Recognition](https://ieeexplore.ieee.org/document/9844653)  
>

**JSTSP**:
1. [Efficient Self-supervised Learning Representations for Spoken Language Identification](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/document/9866521)(test on NIST LRE 2017 & OLR19)  

>This paper illustrates that wav2vec features extracted from XLSR-53 perform well for LID, indicating that features from middle layers (14~16) are the best for LID. In the mean time, a linear bottleneck block (LBN) and an attentive squeeze-and-excitation block are proposed to reduce irrelevant information to improve LID performance. The model with LBN shows even better performance than finetuning. So detailed because this is my work. (Code is available)  


## Performance comparison:
![plot](https://github.com/Lhx94As/Awesome-Spoken-Language-Identification/blob/main/performance_.png)  
**Note that the results of GMM ivector, DNN ivector and xvector are reported in 4.3, and all evaluations are conducted on NIST LRE 07. Some paper used data augmentation which can improve the performance**  
