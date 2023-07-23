
# Awesome Video Generation [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)
## 📝 Introduction
A comprehensive collection of works on video generation/synthesis/prediction.

<br>
<br>

<p align="center">
<img src="https://magvit.cs.cmu.edu/img/fp/00083_gen_sr.gif" width="240px"/>  
<img src="https://magvit.cs.cmu.edu/img/mt_ssv2/full_generation_0143SqueezingSomething_000000782_sr.gif" width="240px"/>
</p>

<p align="center">
<img src="https://modelscope.cn/api/v1/models/damo/text-to-video-synthesis/repo?Revision=master&FilePath=./samples/006_A_cat_eating_food_out_of_a_bowl,_in_style_of_van_Gogh._003.gif&View=true" width="240px"/>  
<img src="https://modelscope.cn/api/v1/models/damo/text-to-video-synthesis/repo?Revision=master&FilePath=./samples/040_Incredibly_detailed_science_fiction_scene_set_on_an_alien_planet,_view_of_a_marketplace._Pixel_art._003.gif&View=true" width="240px"/>
</p>

<p align="center">
<img src="https://kfmei.page/vidm/results/sky/vidm.gif" width="480px"/>  
</p>

<p align="center">
(Source: <a href="https://mask-cond-video-diffusion.github.io">MCVD</a>, <a href="https://modelscope.cn/models/damo/text-to-video-synthesis/summary">VideoFusion</a>, and <a href="https://kfmei.page/vidm/">VIDM</a>)
</p>

## Contents
* [Survey Papers](#survey papers)
* [Datasets](#datasets)
* [Subtopics](#video-generation subtopics)
* [2023](#2023)
* [2022](#2022)
* [2021](#2021)
* [2020](#2020)
* [2019](#2019)
* [2018](#2018)
* [2017](#2017)
* [2016](#2016)

## ✨Survey Papers



+ [Video Frame Interpolation: A Comprehensive Survey](https://dl.acm.org/doi/10.1145/3556544)  

+ [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796)  
  [![Star](https://img.shields.io/github/stars/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy.svg?style=social&label=Star)](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.00796)

+ [Diffusion Models in Vision: A Survey](https://arxiv.org/pdf/2209.04747) (IEEE TPAMI 2023)  
  [![Star](https://img.shields.io/github/stars/CroitoruAlin/Diffusion-Models-in-Vision-A-Survey.svg?style=social&label=Star)](https://github.com/CroitoruAlin/Diffusion-Models-in-Vision-A-Survey)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.04747)

+ [What comprises a good talking-head video generation?: A Survey and Benchmark](https://arxiv.org/pdf/2005.03201)  
  [![Star](https://img.shields.io/github/stars/lelechen63/talking-head-generation-survey.svg?style=social&label=Star)](https://github.com/lelechen63/talking-head-generation-survey)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2005.03201)

+ [A Review on Deep Learning Techniques for Video Prediction](https://arxiv.org/abs/2004.05214) (2020)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2004.05214)

+ [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796)\
  [![Star](https://img.shields.io/github/stars/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy.svg?style=social&label=Star)](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.00796)



## 🌟Datasets



+ [CelebV-Text: A Large-Scale Facial Text-Video Dataset](https://arxiv.org/abs/2303.14717)  
  [![Star](https://img.shields.io/github/stars/CelebV-Text/CelebV-Text.svg?style=social&label=Star)](https://github.com/CelebV-Text/CelebV-Text)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14717)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://celebv-text.github.io/)

+ [CelebV-HQ: A Large-Scale Video Facial Attributes Dataset](https://arxiv.org/abs/2207.12393)  
  [![Star](https://img.shields.io/github/stars/celebv-hq/celebv-hq.svg?style=social&label=Star)](https://github.com/celebv-hq/celebv-hq)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.12393)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://celebv-hq.github.io/)

+ [UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/pdf/1212.0402)  
  [![Star](https://img.shields.io/github/stars/wushidonguc/two-stream-action-recognition-keras.svg?style=social&label=Star)](https://github.com/wushidonguc/two-stream-action-recognition-keras)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1212.0402)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.crcv.ucf.edu/data/UCF101.php)

+ [The Kinetics Human Action Video Dataset](https://arxiv.org/pdf/1705.06950)  
  [![Star](https://img.shields.io/github/stars/deepmind/kinetics-i3d.svg?style=social&label=Star)](https://github.com/deepmind/kinetics-i3d)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1705.06950)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.deepmind.com/open-source/kinetics)

+ [Recognizing human actions: a local SVM approach](https://ieeexplore.ieee.org/document/1334462)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.csc.kth.se/cvap/actions/)

+ [A Short Note about Kinetics-600](https://arxiv.org/pdf/1808.01340)  
  [![Star](https://img.shields.io/github/stars/rocksyne/kinetics-dataset-downloader.svg?style=social&label=Star)](https://github.com/rocksyne/kinetics-dataset-downloader)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1808.01340)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.deepmind.com/open-source/kinetics)

+ [LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs](https://arxiv.org/pdf/2111.02114)  
  [![Star](https://img.shields.io/github/stars/compvis/latent-diffusion.svg?style=social&label=Star)](https://github.com/compvis/latent-diffusion)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.02114)

+ [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/pdf/2104.00650) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/m-bain/frozen-in-time.svg?style=social&label=Star)](https://github.com/m-bain/frozen-in-time)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.00650)

+ [Self-Supervised Visual Planning with Temporal Skip Connections](https://arxiv.org/pdf/1710.05268)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1710.05268)

+ [How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language](https://arxiv.org/pdf/2008.08143) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/how2sign/how2sign.github.io.svg?style=social&label=Star)](https://github.com/how2sign/how2sign.github.io)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2008.08143)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://how2sign.github.io/)

+ [Learning to Drive by Watching YouTube Videos: Action-Conditioned Contrastive Policy Pretraining](https://arxiv.org/pdf/2204.02393) (ECCV 2022)  
  [![Star](https://img.shields.io/github/stars/metadriverse/aco.svg?style=social&label=Star)](https://github.com/metadriverse/aco)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.02393)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://metadriverse.github.io/ACO/)

+ [FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals](https://arxiv.org/pdf/1901.02212)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1901.02212)

+ [DTVNet: Dynamic Time-lapse Video Generation via Single Still Image](https://arxiv.org/pdf/2008.04776) (ECCV 2020)  
  [![Star](https://img.shields.io/github/stars/zhangzjn/DTVNet.svg?style=social&label=Star)](https://github.com/zhangzjn/DTVNet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2008.04776)

+ [Multi-StyleGAN: Towards Image-Based Simulation of Time-Lapse Live-Cell Microscopy](https://arxiv.org/pdf/2106.08285)  
  [![Star](https://img.shields.io/github/stars/ChristophReich1996/Multi-StyleGAN.svg?style=social&label=Star)](https://github.com/ChristophReich1996/Multi-StyleGAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.08285)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://christophreich1996.github.io/multi_stylegan/)

+ [DDH-QA: A Dynamic Digital Humans Quality Assessment Database](https://arxiv.org/abs/2212.12734)\
  [![Star](https://img.shields.io/github/stars/zzc-1998/Point-cloud-quality-assessment.svg?style=social&label=Star)](https://github.com/zzc-1998/Point-cloud-quality-assessment)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.12734)

+ [Muscles in Action](https://arxiv.org/abs/2212.02978)\
  [![Star](https://img.shields.io/github/stars/mchiquier/musclesinactionofficial.svg?style=social&label=Star)](https://github.com/mchiquier/musclesinactionofficial)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.02978)

+ [TPA-Net: Generate A Dataset for Text to Physics-based Animation](https://arxiv.org/abs/2211.13887)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.13887)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/tpa-net)

+ [Touch and Go: Learning from Human-Collected Vision and Touch](https://arxiv.org/abs/2211.12498)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.12498)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://touch-and-go.github.io/)

+ [BVI-VFI: A Video Quality Database for Video Frame Interpolation](https://arxiv.org/abs/2210.00823)\
  [![Star](https://img.shields.io/github/stars/danier97/BVI-VFI-database.svg?style=social&label=Star)](https://github.com/danier97/BVI-VFI-database)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.00823)

+ [Multi-modal Video Chapter Generation](https://arxiv.org/abs/2209.12694)\
  [![Star](https://img.shields.io/github/stars/czt117/MVCG.svg?style=social&label=Star)](https://github.com/czt117/MVCG)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.12694)

+ [Merkel Podcast Corpus: A Multimodal Dataset Compiled from 16 Years of Angela Merkel's Weekly Video Podcasts](https://arxiv.org/abs/2205.12194) (LREC 2022)\
  [![Star](https://img.shields.io/github/stars/deeplsd/Merkel-Podcast-Corpus.svg?style=social&label=Star)](https://github.com/deeplsd/Merkel-Podcast-Corpus)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.12194)

## 🚀Video-generation subtopics

+ [Controllable Video generation]()  
  Text-to-video, image-to-video  
  The “classic” task: create a video scratch, i.e. starting from random noise. The generation process is sometimes given simple conditions, such as Text or image. Common goals include visual fidelity, temporal coherence, and logical plausibility.
+ [Video prediction]()  
  Video generation with (visual) constraints  
Predict the next N frames following a sequence of input video frames, or predict N frames between the given start and final frames.
+ [Frame interpolation]()  
  Could be viewed as a special case of video completion  
  Aimed at improving the motion smoothness of low frame rate videos, by inserting additional frames between existing video frames. Some works can “insert” frames after the input frames, so they technically can perform video prediction to some extent.
+ [Novel view synthesis]()  
  These usually involve reconstructing a 3D scene from some observations (e.g. monocular video input, or static images), and then generating renderings of the scene from new perspectives.
+ [Human motion generation]()  
  These are video generation tasks specifically geared to human (or humanoid) activities
+ [Talking head or face generation]()  
  Talking head generation refers to the generation of animated video content that simulates a person's face and head movements while they are speaking
+ [Video-to-video]()  
  These include enhancing the (textural) quality of videos, style transfer, motion transfer, Summarization, and various common video editing tasks (e.g. removal of a subject).
  

## 2023

+ [Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation](https://arxiv.org/abs/2307.06940)
  [![Star](https://img.shields.io/github/stars/videocrafter/animate-a-story.svg?style=social&label=Star)](https://github.com/videocrafter/animate-a-story)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.06940)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://videocrafter.github.io/Animate-A-Story/)

+ [Probabilistic Adaptation of Text-to-Video Models](https://arxiv.org/abs/2306.01872)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01872)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://video-adapter.github.io/video-adapter/)

+ [VIDM: Video Implicit Diffusion Models](https://arxiv.org/abs/2212.00235) (AAAI 2023)  
  [![Star](https://img.shields.io/github/stars/MKFMIKU/VIDM.svg?style=social&label=Star)](https://github.com/MKFMIKU/VIDM)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.00235)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://kfmei.page/vidm/)

+ [Mm-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation](https://arxiv.org/abs/2212.09478) (CVPR 2023)  
  [![Star](https://img.shields.io/github/stars/researchmm/MM-Diffusion.svg?style=social&label=Star)](https://github.com/researchmm/MM-Diffusion)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.09478)

+ [Video Probabilistic Diffusion Models in Projected Latent Space](https://arxiv.org/abs/2302.07685) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.07685)
  [![Star](https://img.shields.io/github/stars/sihyun-yu/PVDM.svg?style=social&label=Star)](https://github.com/sihyun-yu/PVDM)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sihyun.me/PVDM/)

+ [Synthesizing Artistic Cinemagraphs from Text](https://arxiv.org/abs/2307.03190)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.03190)
  [![Star](https://img.shields.io/github/stars/text2cinemagraph/text2cinemagraph.svg?style=social&label=Star)](https://github.com/text2cinemagraph/text2cinemagraph)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://text2cinemagraph.github.io/website/)

+ [Bidirectional Temporal Diffusion Model for Temporally Consistent Human Animation](https://arxiv.org/abs/2307.00574)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.00574)

+ [DisCo: Disentangled Control for Referring Human Dance Generation in Real World](https://arxiv.org/abs/2307.00040)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.00040)
  [![Star](https://img.shields.io/github/stars/Wangt-CN/DisCo.svg?style=social&label=Star)](https://github.com/Wangt-CN/DisCo)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://disco-dance.github.io/)

+ [PVP: Personalized Video Prior for Editable Dynamic Portraits using StyleGAN](https://arxiv.org/abs/2306.17123)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.17123)
  [![Star](https://img.shields.io/github/stars/ken2576/pvp.svg?style=social&label=Star)](https://github.com/ken2576/pvp)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://cseweb.ucsd.edu//~viscomp/projects/EGSR23PVP/)

+ [BEDLAM: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion](https://arxiv.org/abs/2306.16940) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.16940)
  [![Star](https://img.shields.io/github/stars/pixelite1201/BEDLAM.svg?style=social&label=Star)](https://github.com/pixelite1201/BEDLAM)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://bedlam.is.tue.mpg.de/)

+ [Envisioning a Next Generation Extended Reality Conferencing System with Efficient Photorealistic Human Rendering](https://arxiv.org/abs/2306.16541) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.16541)

+ [Reprogramming Audio-driven Talking Face Synthesis into Text-driven](https://arxiv.org/abs/2306.16003)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.16003)

+ [Self-supervised Learning of Event-guided Video Frame Interpolation for Rolling Shutter Frames](https://arxiv.org/abs/2306.15507)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.15507)

+ [Boost Video Frame Interpolation via Motion Adaptation](https://arxiv.org/abs/2306.13933)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.13933)

+ [VidEdit: Zero-Shot and Spatially Aware Text-Driven Video Editing](https://arxiv.org/abs/2306.08707)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08707)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://videdit.github.io/)

+ [DORSal: Diffusion for Object-centric Representations of Scenes](https://arxiv.org/abs/2306.08068)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08068)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.sjoerdvansteenkiste.com/dorsal/)

+ [Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation](https://arxiv.org/abs/2306.07954)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.07954)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://anonymous-31415926.github.io/)

+ [Generative Semantic Communication: Diffusion Models Beyond Bit Recovery](https://arxiv.org/abs/2306.04321)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.04321)

+ [Instruct-Video2Avatar: Video-to-Avatar Generation with Instructions](https://arxiv.org/abs/2306.02903)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02903)
  [![Star](https://img.shields.io/github/stars/lsx0101/Instruct-Video2Avatar.svg?style=social&label=Star)](https://github.com/lsx0101/Instruct-Video2Avatar)

+ [MoviePuzzle: Visual Narrative Reasoning through Multimodal Order Learning](https://arxiv.org/abs/2306.02252)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02252)

+ [Video Colorization with Pre-trained Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.01732)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01732)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://colordiffuser.github.io/)

+ [Temporal-controlled Frame Swap for Generating High-Fidelity Stereo Driving Data for Autonomy Analysis](https://arxiv.org/abs/2306.01704)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01704)
  [![Star](https://img.shields.io/github/stars/ostadabbas/Temporal-controlled-Frame-Swap-GTAV-TeFS-.svg?style=social&label=Star)](https://github.com/ostadabbas/Temporal-controlled-Frame-Swap-GTAV-TeFS-)

+ [Adjustable Visual Appearance for Generalizable Novel View Synthesis](https://arxiv.org/abs/2306.01344)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01344)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://ava-nvs.github.io/)

+ [4DSR-GCN: 4D Video Point Cloud Upsampling using Graph Convolutional Networks](https://arxiv.org/abs/2306.01081)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01081)

+ [Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models](https://arxiv.org/abs/2306.00973)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.00973)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://haoningwu3639.github.io/StoryGen_Webpage/)

+ [MammalNet: A Large-scale Video Benchmark for Mammal Recognition and Behavior Understanding](https://arxiv.org/abs/2306.00576)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.00576)

+ [Exploring Phonetic Context in Lip Movement for Authentic Talking Face Generation](https://arxiv.org/abs/2305.19556)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.19556)

+ [Video ControlNet: Towards Temporally Consistent Synthetic-to-Real Video Translation Using Conditional Image Diffusion Models](https://arxiv.org/abs/2305.19193)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.19193)

+ [Context-Preserving Two-Stage Video Domain Translation for Portrait Stylization](https://arxiv.org/abs/2305.19135)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.19135)

+ [OD-NeRF: Efficient Training of On-the-Fly Dynamic Neural Radiance Fields](https://arxiv.org/abs/2305.14831)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14831)

+ [EgoVSR: Towards High-Quality Egocentric Video Super-Resolution](https://arxiv.org/abs/2305.14708)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14708)

+ [NegVSR: Augmenting Negatives for Generalized Noise Modeling in Real-World Video Super-Resolution](https://arxiv.org/abs/2305.14669)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14669)

+ [Video Prediction Models as Rewards for Reinforcement Learning](https://arxiv.org/abs/2305.14343)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14343)

+ [Reparo: Loss-Resilient Generative Codec for Video Conferencing](https://arxiv.org/abs/2305.14135)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14135)

+ [CPNet: Exploiting CLIP-based Attention Condenser and Probability Map Guidance for High-fidelity Talking Face Generation](https://arxiv.org/abs/2305.13962) (ICME 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.13962)

+ [Synthesizing Diverse Human Motions in 3D Indoor Scenes](https://arxiv.org/abs/2305.12411)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.12411)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://zkf1997.github.io/DIMOS/)

+ [InstructVid2Vid: Controllable Video Editing with Natural Language Instructions](https://arxiv.org/abs/2305.12328)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.12328)

+ [SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models](https://arxiv.org/abs/2305.11281) (ICLR Workshop 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.11281)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://slotdiffusion.github.io/)

+ [IDO-VFI: Identifying Dynamics via Optical Flow Guidance for Video Frame Interpolation with Events](https://arxiv.org/abs/2305.10198)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10198)

+ [Light-VQA: A Multi-Dimensional Quality Assessment Model for Low-Light Video Enhancement](https://arxiv.org/abs/2305.09512)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.09512)

+ [Laughing Matters: Introducing Laughing-Face Generation using Diffusion Models](https://arxiv.org/abs/2305.08854)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.08854)

+ [Identity-Preserving Talking Face Generation with Landmark and Appearance Priors](https://arxiv.org/abs/2305.08293) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.08293)
  [![Star](https://img.shields.io/github/stars/Weizhi-Zhong/IP_LAP.svg?style=social&label=Star)](https://github.com/Weizhi-Zhong/IP_LAP)

+ [HumanRF: High-Fidelity Neural Radiance Fields for Humans in Motion](https://arxiv.org/abs/2305.06356)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.06356)
  [![Star](https://img.shields.io/github/stars/synthesiaresearch/humanrf.svg?style=social&label=Star)](https://github.com/synthesiaresearch/humanrf)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://synthesiaresearch.github.io/humanrf/)

+ [Style-A-Video: Agile Diffusion for Arbitrary Text-based Video Style Transfer](https://arxiv.org/abs/2305.05464)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.05464)
  [![Star](https://img.shields.io/github/stars/haha-lisa/Style-A-Video.svg?style=social&label=Star)](https://github.com/haha-lisa/Style-A-Video)

+ [NeuralEditor: Editing Neural Radiance Fields via Manipulating Point Clouds](https://arxiv.org/abs/2305.03049) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.03049)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://immortalco.github.io/NeuralEditor/)

+ [DSEC-MOS: Segment Any Moving Object with Moving Ego Vehicle](https://arxiv.org/abs/2305.00126)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.00126)
  [![Star](https://img.shields.io/github/stars/ZZY-Zhou/DSEC-MOS.svg?style=social&label=Star)](https://github.com/ZZY-Zhou/DSEC-MOS)

+ [Video Frame Interpolation with Densely Queried Bilateral Correlation](https://arxiv.org/abs/2304.13596) (IJCAI 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.13596)
  [![Star](https://img.shields.io/github/stars/kinoud/DQBC.svg?style=social&label=Star)](https://github.com/kinoud/DQBC)

+ [Dynamic Video Frame Interpolation with integrated Difficulty Pre-Assessment](https://arxiv.org/abs/2304.12664)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.12664)

+ [AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation](https://arxiv.org/abs/2304.09790) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.09790)
  [![Star](https://img.shields.io/github/stars/MCG-NKU/AMT.svg?style=social&label=Star)](https://github.com/MCG-NKU/AMT)

+ [Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation](https://arxiv.org/abs/2304.08477)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.08477)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://latent-shift.github.io/)

+ [CAT-NeRF: Constancy-Aware Tx$^2$Former for Dynamic Body Modeling](https://arxiv.org/abs/2304.07915) (CVPR Workshop 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.07915)

+ [Soundini: Sound-Guided Diffusion for Natural Video Editing](https://arxiv.org/abs/2304.06818)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.06818)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://kuai-lab.github.io/soundini-gallery/)

+ [Boosting Video Object Segmentation via Space-time Correspondence Learning](https://arxiv.org/abs/2304.06211) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.06211)
  [![Star](https://img.shields.io/github/stars/wenguanwang/VOS_Correspondence.svg?style=social&label=Star)](https://github.com/wenguanwang/VOS_Correspondence)

+ [VidStyleODE: Disentangled Video Editing via StyleGAN and NeuralODEs](https://arxiv.org/abs/2304.06020)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.06020)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://cyberiada.github.io/VidStyleODE/)

+ [MED-VT: Multiscale Encoder-Decoder Video Transformer with Application to Object Segmentation](https://arxiv.org/abs/2304.05930) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.05930)
  [![Star](https://img.shields.io/github/stars/rkyuca/medvt.svg?style=social&label=Star)](https://github.com/rkyuca/medvt)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://rkyuca.github.io/medvt/)

+ [Neural Image-based Avatars: Generalizable Radiance Fields for Human Avatar Modeling](https://arxiv.org/abs/2304.04897)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.04897)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://youngjoongunc.github.io/nia/)

+ [That's What I Said: Fully-Controllable Talking Face Generation](https://arxiv.org/abs/2304.03275)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.03275)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://mm.kaist.ac.kr/projects/FC-TFG/)

+ [HNeRV: A Hybrid Neural Representation for Videos](https://arxiv.org/abs/2304.02633) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.02633)
  [![Star](https://img.shields.io/github/stars/haochen-rye/HNeRV.svg?style=social&label=Star)](https://github.com/haochen-rye/HNeRV)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://haochen-rye.github.io/HNeRV/)

+ [BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation](https://arxiv.org/abs/2304.02225) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.02225)
  [![Star](https://img.shields.io/github/stars/JunHeum/BiFormer.svg?style=social&label=Star)](https://github.com/JunHeum/BiFormer)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://openaccess.thecvf.com/content/CVPR2023/papers/Park_BiFormer_Learning_Bilateral_Motion_Estimation_via_Bilateral_Transformer_for_4K_CVPR_2023_paper.pdf)

+ [TalkCLIP: Talking Head Generation with Text-Guided Expressive Speaking Styles](https://arxiv.org/abs/2304.00334)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.00334)

+ [FONT: Flow-guided One-shot Talking Head Generation with Natural Head Motions](https://arxiv.org/abs/2303.17789) (ICME 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.17789)

+ [Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models](https://arxiv.org/abs/2303.17599)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.17599)
  [![Star](https://img.shields.io/github/stars/baaivision/vid2vid-zero.svg?style=social&label=Star)](https://github.com/baaivision/vid2vid-zero)

+ [Consistent View Synthesis with Pose-Guided Diffusion Models](https://arxiv.org/abs/2303.17598) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.17598)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://poseguided-diffusion.github.io/)

+ [DAE-Talker: High Fidelity Speech-Driven Talking Face Generation with Diffusion Autoencoder](https://arxiv.org/abs/2303.17550)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.17550)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://daetalker.github.io/)

+ [Novel View Synthesis of Humans using Differentiable Rendering](https://arxiv.org/abs/2303.15880)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.15880)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://github.com/GuillaumeRochette/HumanViewSynthesis)

+ [CelebV-Text: A Large-Scale Facial Text-Video Dataset](https://arxiv.org/abs/2303.14717) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14717)
  [![Star](https://img.shields.io/github/stars/celebv-text/CelebV-Text.svg?style=social&label=Star)](https://github.com/celebv-text/CelebV-Text)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://celebv-text.github.io/)

+ [GestureDiffuCLIP: Gesture Diffusion Model with CLIP Latents](https://arxiv.org/abs/2303.14613) (SIGGRAPH 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14613)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://pku-mocca.github.io/GestureDiffuCLIP-Page/)

+ [SUDS: Scalable Urban Dynamic Scenes](https://arxiv.org/abs/2303.14536) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14536)
  [![Star](https://img.shields.io/github/stars/hturki/suds.svg?style=social&label=Star)](https://github.com/hturki/suds)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://haithemturki.com/suds/)

+ [NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects](https://arxiv.org/abs/2303.14435) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14435)
  [![Star](https://img.shields.io/github/stars/JokerYan/NeRF-DS.svg?style=social&label=Star)](https://github.com/JokerYan/NeRF-DS)

+ [HandNeRF: Neural Radiance Fields for Animatable Interacting Hands](https://arxiv.org/abs/2303.13825) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.13825)

+ [Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.13439)
  [![Star](https://img.shields.io/github/stars/Picsart-AI-Research/Text2Video-Zero.svg?style=social&label=Star)](https://github.com/Picsart-AI-Research/Text2Video-Zero)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://text2video-zero.github.io/)

+ [SHERF: Generalizable Human NeRF from a Single Image](https://arxiv.org/abs/2303.12791)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.12791)
  [![Star](https://img.shields.io/github/stars/skhu101/SHERF.svg?style=social&label=Star)](https://github.com/skhu101/SHERF)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://skhu101.github.io/SHERF/)

+ [Pix2Video: Video Editing using Image Diffusion](https://arxiv.org/abs/2303.12688)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.12688)
  [![Star](https://img.shields.io/github/stars/G-U-N/Pix2Video.pytorch.svg?style=social&label=Star)](https://github.com/G-U-N/Pix2Video.pytorch)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://duyguceylan.github.io/pix2video.github.io/)

+ [Music-Driven Group Choreography](https://arxiv.org/abs/2303.12337) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.12337)
  [![Star](https://img.shields.io/github/stars/aioz-ai/AIOZ-GDANCE.svg?style=social&label=Star)](https://github.com/aioz-ai/AIOZ-GDANCE)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://aioz-ai.github.io/AIOZ-GDANCE/)

+ [Pre-NeRF 360: Enriching Unbounded Appearances for Neural Radiance Fields](https://arxiv.org/abs/2303.12234)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.12234)
  [![Star](https://img.shields.io/github/stars/CVUBLab/pre-nerf.svg?style=social&label=Star)](https://github.com/CVUBLab/pre-nerf)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://amughrabi.github.io/prenerf/)

+ [Emotionally Enhanced Talking Face Generation](https://arxiv.org/abs/2303.11548)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.11548)
  [![Star](https://img.shields.io/github/stars/sahilg06/EmoGen.svg?style=social&label=Star)](https://github.com/sahilg06/EmoGen)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://midas.iiitd.edu.in/emo/)

+ [Tubelet-Contrastive Self-Supervision for Video-Efficient Generalization](https://arxiv.org/abs/2303.11003)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.11003)

+ [Confidence Attention and Generalization Enhanced Distillation for Continuous Video Domain Adaptation](https://arxiv.org/abs/2303.10452)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.10452)

+ [MoRF: Mobile Realistic Fullbody Avatars from a Monocular Video](https://arxiv.org/abs/2303.10275)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.10275)

+ [Unified Mask Embedding and Correspondence Learning for Self-Supervised Video Segmentation](https://arxiv.org/abs/2303.10100)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.10100)

+ [Leaping Into Memories: Space-Time Deep Feature Synthesis](https://arxiv.org/abs/2303.09941)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09941)

+ [Learning Data-Driven Vector-Quantized Degradation Model for Animation Video Super-Resolution](https://arxiv.org/abs/2303.09826)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09826)

+ [FateZero: Fusing Attentions for Zero-shot Text-based Video Editing](https://arxiv.org/abs/2303.09535)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09535)
  [![Star](https://img.shields.io/github/stars/ChenyangQiQi/FateZero.svg?style=social&label=Star)](https://github.com/ChenyangQiQi/FateZero)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://fate-zero-edit.github.io/)

+ [LDMVFI: Video Frame Interpolation with Latent Diffusion Models](https://arxiv.org/abs/2303.09508)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09508)

+ [Learning Physical-Spatio-Temporal Features for Video Shadow Removal](https://arxiv.org/abs/2303.09370)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09370)

+ [NLUT: Neural-based 3D Lookup Tables for Video Photorealistic Style Transfer](https://arxiv.org/abs/2303.09170)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09170)
  [![Star](https://img.shields.io/github/stars/semchan/NLUT.svg?style=social&label=Star)](https://github.com/semchan/NLUT)

+ [Blowing in the Wind: CycleNet for Human Cinemagraphs from Still Images](https://arxiv.org/abs/2303.08639) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.08639)
  [![Star](https://img.shields.io/github/stars/CycleNet/.svg?style=social&label=Star)](https://hbertiche.github.io/CycleNet/)

+ [Blind Video Deflickering by Neural Filtering with a Flawed Atlas](https://arxiv.org/abs/2303.08120) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.08120)
  [![Star](https://img.shields.io/github/stars/ChenyangLEI/All-In-One-Deflicker.svg?style=social&label=Star)](https://github.com/ChenyangLEI/All-In-One-Deflicker)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://chenyanglei.github.io/deflicker/)

+ [Butterfly: Multiple Reference Frames Feature Propagation Mechanism for Neural Video Compression](https://arxiv.org/abs/2303.02959) (DCC 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.02959)

+ [One-Shot Video Inpainting](https://arxiv.org/abs/2302.14362) (AAAI 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.14362)

+ [Continuous Space-Time Video Super-Resolution Utilizing Long-Range Temporal Information](https://arxiv.org/abs/2302.13256)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.13256)

+ [Learning Neural Volumetric Representations of Dynamic Humans in Minutes](https://arxiv.org/abs/2302.12237) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.12237)
  [![Star](https://img.shields.io/github/stars/zju3dv/instant-nvr.svg?style=social&label=Star)](https://github.com/zju3dv/instant-nvr)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://zju3dv.github.io/instant_nvr/)

+ [STB-VMM: Swin Transformer Based Video Motion Magnification](https://arxiv.org/abs/2302.10001)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.10001)
  [![Star](https://img.shields.io/github/stars/RLado/STB-VMM.svg?style=social&label=Star)](https://github.com/RLado/STB-VMM)

+ [OPT: One-shot Pose-Controllable Talking Head Generation](https://arxiv.org/abs/2302.08197) (ICASSP 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.08197)

+ [One-Shot Face Video Re-enactment using Hybrid Latent Spaces of StyleGAN2](https://arxiv.org/abs/2302.07848)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.07848)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://trevineoorloff.github.io/FaceVideoReenactment_HybridLatents.io/)

+ [Video Waterdrop Removal via Spatio-Temporal Fusion in Driving Scenes](https://arxiv.org/abs/2302.05916)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.05916)
  [![Star](https://img.shields.io/github/stars/csqiangwen/Video_Waterdrop_Removal_in_Driving_Scenes.svg?style=social&label=Star)](https://github.com/csqiangwen/Video_Waterdrop_Removal_in_Driving_Scenes)

+ [Structure and Content-Guided Video Synthesis with Diffusion Models](https://arxiv.org/abs/2302.03011)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.03011)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://research.runwayml.com/gen1)

+ [AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis](https://arxiv.org/abs/2302.02088)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.02088)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://liangsusan-git.github.io/project/avnerf/)

+ [Dreamix: Video Diffusion Models are General Video Editors](https://arxiv.org/abs/2302.01329)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.01329)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://dreamix-video-editing.github.io/)

+ [SceneScape: Text-Driven Consistent Scene Generation](https://arxiv.org/abs/2302.01133)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.01133)
  [![Star](https://img.shields.io/github/stars/RafailFridman/SceneScape.svg?style=social&label=Star)](https://github.com/RafailFridman/SceneScape)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://scenescape.github.io/)

+ [Maximal Cliques on Multi-Frame Proposal Graph for Unsupervised Video Object Segmentation](https://arxiv.org/abs/2301.12352)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.12352)

+ [Optical Flow Estimation in 360$^\circ$ Videos: Dataset, Model and Application](https://arxiv.org/abs/2301.11880)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.11880)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://siamlof.github.io/)

+ [Unsupervised Volumetric Animation](https://arxiv.org/abs/2301.11326)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.11326)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://snap-research.github.io/unsupervised-volumetric-animation/)

+ [Text-To-4D Dynamic Scene Generation](https://arxiv.org/abs/2301.11280)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.11280)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://make-a-video3d.github.io/)

+ [Regeneration Learning: A Learning Paradigm for Data Generation](https://arxiv.org/abs/2301.08846)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.08846)

+ [Event-Based Frame Interpolation with Ad-hoc Deblurring](https://arxiv.org/abs/2301.05191) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.05191)
  [![Star](https://img.shields.io/github/stars/AHupuJR/REFID.svg?style=social&label=Star)](https://github.com/AHupuJR/REFID)

+ [DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation](https://arxiv.org/abs/2301.03786) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.03786)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sstzal.github.io/DiffTalk/)

+ [Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation](https://arxiv.org/abs/2301.03396)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.03396)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://mstypulkowski.github.io/diffusedheads/)

+ [HyperReel: High-Fidelity 6-DoF Video with Ray-Conditioned Sampling](https://arxiv.org/abs/2301.02238) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.02238)
  [![Star](https://img.shields.io/github/stars/facebookresearch/hyperreel.svg?style=social&label=Star)](https://github.com/facebookresearch/hyperreel)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://hyperreel.github.io/)

+ [StyleTalk: One-shot Talking Head Generation with Controllable Speaking Styles](https://arxiv.org/abs/2301.01081) (AAAI 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.01081)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://github.com/FuxiVirtualHuman/styletalk)

+ [Detachable Novel Views Synthesis of Dynamic Scenes Using Distribution-Driven Neural Radiance Fields](https://arxiv.org/abs/2301.00411)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.00411)
  [![Star](https://img.shields.io/github/stars/Luciferbobo/D4NeRF.svg?style=social&label=Star)](https://github.com/Luciferbobo/D4NeRF)

+ [SkyGPT: Probabilistic Short-term Solar Forecasting Using Synthetic Sky Videos from Physics-constrained VideoGPT](https://arxiv.org/abs/2306.11682)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.11682)

+ [MovieFactory: Automatic Movie Creation from Text using Large Generative Models for Language and Images](https://arxiv.org/abs/2306.07257)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.07257)

+ [Emotional Talking Head Generation based on Memory-Sharing and Attention-Augmented Networks](https://arxiv.org/abs/2306.03594)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.03594)

+ [Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes](https://arxiv.org/abs/2305.11772)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.11772)

+ [Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](https://arxiv.org/abs/2305.03713)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.03713)

+ [DynamicStereo: Consistent Dynamic Depth from Stereo Videos](https://arxiv.org/abs/2305.02296) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.02296)
  [![Star](https://img.shields.io/github/stars/facebookresearch/dynamic_stereo.svg?style=social&label=Star)](https://github.com/facebookresearch/dynamic_stereo)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://dynamic-stereo.github.io/)

+ [ActorsNeRF: Animatable Few-shot Human Rendering with Generalizable NeRFs](https://arxiv.org/abs/2304.14401)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.14401)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://jitengmu.github.io/ActorsNeRF/)

+ [Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis](https://arxiv.org/abs/2304.12317)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.12317)
  [![Star](https://img.shields.io/github/stars/andrewsonga/Total-Recon.svg?style=social&label=Star)](https://github.com/andrewsonga/Total-Recon)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://andrewsonga.github.io/totalrecon/)

+ [3D-IntPhys: Towards More Generalized 3D-grounded Visual Intuitive Physics under Challenging Scenes](https://arxiv.org/abs/2304.11470) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.11470)

+ [Leveraging triplet loss for unsupervised action segmentation](https://arxiv.org/abs/2304.06403) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.06403)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://github.com/elenabbbuenob/tsa-actionseg)

+ [MonoHuman: Animatable Human Neural Field from Monocular Video](https://arxiv.org/abs/2304.02001) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.02001)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yzmblog.github.io/projects/MonoHuman/)

+ [Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion](https://arxiv.org/abs/2304.01893) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.01893)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://research.nvidia.com/labs/toronto-ai/trace-pace/)

+ [Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert](https://arxiv.org/abs/2303.17480) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.17480)
  [![Star](https://img.shields.io/github/stars/Sxjdwang/TalkLip.svg?style=social&label=Star)](https://github.com/Sxjdwang/TalkLip)

+ [VIVE3D: Viewpoint-Independent Video Editing using 3D-Aware GANs](https://arxiv.org/abs/2303.15893) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.15893)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://afruehstueck.github.io/vive3D/)

+ [CAMS: CAnonicalized Manipulation Spaces for Category-Level Functional Hand-Object Manipulation Synthesis](https://arxiv.org/abs/2303.15469) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.15469)
  [![Star](https://img.shields.io/github/stars/cams-hoi/CAMS.svg?style=social&label=Star)](https://github.com/cams-hoi/CAMS)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://cams-hoi.github.io/)

+ [Prediction of the morphological evolution of a splashing drop using an encoder-decoder](https://arxiv.org/abs/2303.14109)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14109)

+ [TriPlaneNet: An Encoder for EG3D Inversion](https://arxiv.org/abs/2303.13497)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.13497)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://anantarb.github.io/triplanenet)

+ [Dual-path Adaptation from Image to Video Transformers](https://arxiv.org/abs/2303.09857) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09857)
  [![Star](https://img.shields.io/github/stars/park-jungin/DualPath.svg?style=social&label=Star)](https://github.com/park-jungin/DualPath)

+ [Video-P2P: Video Editing with Cross-attention Control](https://arxiv.org/abs/2303.04761)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.04761)
  [![Star](https://img.shields.io/github/stars/ShaoTengLiu/Video-P2P.svg?style=social&label=Star)](https://github.com/ShaoTengLiu/Video-P2P)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://video-p2p.github.io/)

+ [IntrinsicNGP: Intrinsic Coordinate based Hash Encoding for Human NeRF](https://arxiv.org/abs/2302.14683)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.14683)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://ustc3dv.github.io/IntrinsicNGP/)

+ [Robust Dynamic Radiance Fields](https://arxiv.org/abs/2301.02239) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.02239)
  [![Star](https://img.shields.io/github/stars/facebookresearch/robust-dynrf.svg?style=social&label=Star)](https://github.com/facebookresearch/robust-dynrf)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://robust-dynrf.github.io/)



## 2022



+ [Video Diffusion Models](https://arxiv.org/abs/2204.03458) (NeurIPS 2022)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.03458)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://video-diffusion.github.io/)

+ [McVd: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation](https://arxiv.org/abs/2205.09853) (NeurIPS 2022)  
  [![Star](https://img.shields.io/github/stars/Tobi-r9/RaMViD.svg?style=social&label=Star)](https://github.com/voletiv/mcvd-pytorch)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.09853)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://mask-cond-video-diffusion.github.io)

+ [Diffusion Models for Video Prediction and Infilling](https://arxiv.org/abs/2206.07696) (TMLR 2022)  
  [![Star](https://img.shields.io/github/stars/Tobi-r9/RaMViD.svg?style=social&label=Star)](https://github.com/Tobi-r9/RaMViD)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.07696)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/video-diffusion-prediction)

+ [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://openreview.net/forum?id=nJfylDvgzlq) (ICLR 2023)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://openreview.net/forum?id=nJfylDvgzlq)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://makeavideo.studio)

+ [DaGAN: Depth-Aware Generative Adversarial Network for Talking Head Video Generation](https://arxiv.org/abs/2203.06605) (CVPR 2022)  
  [![Star](https://img.shields.io/github/stars/harlanhong/CVPR2022-DaGAN.svg?style=social&label=Star)](https://github.com/harlanhong/CVPR2022-DaGAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.06605)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://harlanhong.github.io/publications/dagan.html)

+ [Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning](https://arxiv.org/abs/2203.02573) (CVPR 2022)  
  [![Star](https://img.shields.io/github/stars/snap-research/MMVID.svg?style=social&label=Star)](https://github.com/snap-research/MMVID)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.02573)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://snap-research.github.io/MMVID/)

+ [Playable Environments: Video Manipulation in Space and Time](https://arxiv.org/abs/2203.01914) (CVPR 2022)  
  [![Star](https://img.shields.io/github/stars/willi-menapace/PlayableEnvironments.svg?style=social&label=Star)](https://github.com/willi-menapace/PlayableEnvironments)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.01914)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://willi-menapace.github.io/playable-environments-website/)

+ [Fast-Vid2Vid: Spatial-Temporal Compression for Video-to-Video Synthesis](https://arxiv.org/abs/2207.05049) (ECCV 2022)  
  [![Star](https://img.shields.io/github/stars/fast-vid2vid/fast-vid2vid.svg?style=social&label=Star)](https://github.com/fast-vid2vid/fast-vid2vid)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.05049)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://fast-vid2vid.github.io/)

+ [TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts](https://arxiv.org/abs/2207.01696) (ECCV 2022)  
  [![Star](https://img.shields.io/github/stars/EricGuo5513/TM2T.svg?style=social&label=Star)](https://github.com/EricGuo5513/TM2T)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.01696)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://ericguo5513.github.io/TM2T/)

+ [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02303)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://video-diffusion.github.io/)


+ [Phenaki: Variable length video generation from open domain textual description](https://arxiv.org/abs/2210.02399)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02399)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.research.google/phenaki/)
  [alt-website](https://phenaki.video/)

  Code (unofficial?): [![Star](https://img.shields.io/github/stars/lucidrains/phenaki-pytorch.svg?style=social&label=Star)](https://github.com/XX/YY)

+ [Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2212.11565)  
  [![Star](https://img.shields.io/github/stars/bbzhu-jy16/motionvideogan.svg?style=social&label=Star)](https://github.com/showlab/Tune-A-Video)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.11565)

+ [Towards Smooth Video Composition](https://arxiv.org/abs/2212.07413)  
  [![Star](https://img.shields.io/github/stars/genforce/StyleSV.svg?style=social&label=Star)](https://github.com/genforce/StyleSV)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.07413)

+ [Latent Video Diffusion Models for High-Fidelity Long Video Generation](https://arxiv.org/abs/2211.13221)  
  [![Star](https://img.shields.io/github/stars/yingqinghe/lvdm.svg?style=social&label=Star)](https://github.com/yingqinghe/lvdm)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.13221)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yingqinghe.github.io/LVDM/)

+ [SinFusion: Training Diffusion Models on a Single Image or Video](https://arxiv.org/abs/2211.11743)  
  [![Star](https://img.shields.io/github/stars/yanivnik/sinfusion-code.svg?style=social&label=Star)](https://github.com/yanivnik/sinfusion-code)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.11743)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yanivnik.github.io/sinfusion/static/video_comparisons.html)

+ [INR-V: A Continuous Representation Space for Video-based Generative Tasks](https://arxiv.org/abs/2210.16579)  
  [![Star](https://img.shields.io/github/stars/bipashasen/INRV.svg?style=social&label=Star)](https://github.com/bipashasen/INRV)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.16579)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://skymanaditya1.github.io/INRV/)

+ [Computational Choreography using Human Motion Synthesis](https://arxiv.org/abs/2210.04366)  
  [![Star](https://img.shields.io/github/stars/patrickrperrine/comp-choreo.svg?style=social&label=Star)](https://github.com/patrickrperrine/comp-choreo)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.04366)

+ [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399)  
  [![Star](https://img.shields.io/github/stars/lucidrains/phenaki-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/phenaki-pytorch)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02399)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://phenaki.video/)

+ [Temporally Consistent Transformers for Video Generation](https://arxiv.org/abs/2210.02396)  
  [![Star](https://img.shields.io/github/stars/wilson1yan/teco.svg?style=social&label=Star)](https://github.com/wilson1yan/teco)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02396)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://wilson1yan.github.io/teco/)

+ [StyleFaceV: Face Video Generation via Decomposing and Recomposing Pretrained StyleGAN3](https://arxiv.org/abs/2208.07862)  
  [![Star](https://img.shields.io/github/stars/arthur-qiu/stylefacev.svg?style=social&label=Star)](https://github.com/arthur-qiu/stylefacev)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.07862)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://haonanqiu.com/projects/StyleFaceV.html)

+ [NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis](https://arxiv.org/abs/2207.09814)  
  [![Star](https://img.shields.io/github/stars/arthur-qiu/stylefacev.svg?style=social&label=Star)](https://github.com/microsoft/nuwa)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.09814)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://nuwa-infinity.microsoft.com/)

+ [3D-Aware Video Generation](https://arxiv.org/abs/2206.14797)  
  [![Star](https://img.shields.io/github/stars/sherwinbahmani/3dvideogeneration.svg?style=social&label=Star)](https://github.com/sherwinbahmani/3dvideogeneration/)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.14797)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sherwinbahmani.github.io/3dvidgen/)

+ [Patch-based Object-centric Transformers for Efficient Video Generation](https://arxiv.org/abs/2206.04003)  
  [![Star](https://img.shields.io/github/stars/wilson1yan/povt.svg?style=social&label=Star)](https://github.com/wilson1yan/povt)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.04003)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/povt-public)

+ [Generating Long Videos of Dynamic Scenes](https://arxiv.org/abs/2206.03429)  
  [![Star](https://img.shields.io/github/stars/nvlabs/long-video-gan.svg?style=social&label=Star)](https://github.com/nvlabs/long-video-gan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.03429)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.timothybrooks.com/tech/long-video-gan/)

+ [D'ARTAGNAN: Counterfactual Video Generation](https://arxiv.org/abs/2206.01651)  
  [![Star](https://img.shields.io/github/stars/nvlabs/long-video-gan.svg?style=social&label=Star)](https://github.com/hreynaud/dartagnan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.01651)

+ [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)  
  [![Star](https://img.shields.io/github/stars/nvlabs/long-video-gan.svg?style=social&label=Star)](https://github.com/thudm/cogvideo)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.15868)

+ [Latent Video Diffusion Models for High-Fidelity Video Generation With Arbitrary Lengths](https://arxiv.org/abs/2211.13221)  
  [![Star](https://img.shields.io/github/stars/YingqingHe/LVDM.svg?style=social&label=Star)](https://github.com/YingqingHe/LVDM)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.13221)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yingqinghe.github.io/LVDM/)

+ [MagicVideo: Efficient Video Generation With Latent Diffusion Models](https://arxiv.org/abs/2211.11018)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.11018)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://magicvideo.github.io/#)

+ [Diffusion Probabilistic Modeling for Video Generation](https://arxiv.org/abs/2203.09481)  
  [![Star](https://img.shields.io/github/stars/buggyyang/RVD.svg?style=social&label=Star)](https://github.com/buggyyang/RVD)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09481)

+ [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495)  
  [![Star](https://img.shields.io/github/stars/plai-group/flexible-video-diffusion-modeling.svg?style=social&label=Star)](https://github.com/plai-group/flexible-video-diffusion-modeling)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.11495)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://fdmolv.github.io/)

+ [Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer](https://arxiv.org/pdf/2204.03638) (ECCV 2022)  
  [![Star](https://img.shields.io/github/stars/songweige/tats.svg?style=social&label=Star)](https://github.com/songweige/tats)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.03638)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://songweige.github.io/projects/tats/index.html)

+ [Diffusion Probabilistic Modeling for Video Generation](https://arxiv.org/pdf/2203.09481)  
  [![Star](https://img.shields.io/github/stars/buggyyang/rvd.svg?style=social&label=Star)](https://github.com/buggyyang/rvd)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09481)

+ [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN](https://arxiv.org/pdf/2203.04036)  
  [![Star](https://img.shields.io/github/stars/OpenTalker/StyleHEAT.svg?style=social&label=Star)](https://github.com/OpenTalker/StyleHEAT)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.04036)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://feiiyin.github.io/StyleHEAT/)

+ [Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks](https://arxiv.org/pdf/2202.10571) (ICLR 2022)  
  [![Star](https://img.shields.io/github/stars/sihyun-yu/digan.svg?style=social&label=Star)](https://github.com/sihyun-yu/digan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.10571)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sihyun.me/digan/)

+ [StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2](https://arxiv.org/pdf/2112.14683) (CVPR 2022)  
  [![Star](https://img.shields.io/github/stars/universome/stylegan-v.svg?style=social&label=Star)](https://github.com/universome/stylegan-v)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.14683)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://universome.github.io/stylegan-v.html)

+ [Make It Move: Controllable Image-to-Video Generation with Text Descriptions](https://arxiv.org/pdf/2112.02815) (CVPR 2022)  
  [![Star](https://img.shields.io/github/stars/youncy-hu/mage.svg?style=social&label=Star)](https://github.com/youncy-hu/mage)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.02815)

+ [NeMo: 3D Neural Motion Fields from Multiple Video Instances of the Same Action](https://arxiv.org/abs/2212.13660) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/wangkua1/nemo-cvpr2023.svg?style=social&label=Star)](https://github.com/wangkua1/nemo-cvpr2023)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.13660)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/nemo-neural-motion-field)

+ [Cross-Resolution Flow Propagation for Foveated Video Super-Resolution](https://arxiv.org/abs/2212.13525) (WACV 2023)\
  [![Star](https://img.shields.io/github/stars/eugenelet/CRFP.svg?style=social&label=Star)](https://github.com/eugenelet/CRFP)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.13525)

+ [MonoNeRF: Learning a Generalizable Dynamic Radiance Field from Monocular Videos](https://arxiv.org/abs/2212.13056)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.13056)

+ [Scalable Adaptive Computation for Iterative Generation](https://arxiv.org/abs/2212.11972) (ICML 2023)\
  [![Star](https://img.shields.io/github/stars/google-research/pix2seq.svg?style=social&label=Star)](https://github.com/google-research/pix2seq)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.11972)

+ [Predictive Coding Based Multiscale Network with Encoder-Decoder LSTM for Video Prediction](https://arxiv.org/abs/2212.11642)\
  [![Star](https://img.shields.io/github/stars/Ling-CF/MSPN.svg?style=social&label=Star)](https://github.com/Ling-CF/MSPN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.11642)

+ [InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds](https://arxiv.org/abs/2212.10550)\
  [![Star](https://img.shields.io/github/stars/tijiang13/InstantAvatar.svg?style=social&label=Star)](https://github.com/tijiang13/InstantAvatar)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.10550)

+ [MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation](https://arxiv.org/abs/2212.09478) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/researchmm/MM-Diffusion.svg?style=social&label=Star)](https://github.com/researchmm/MM-Diffusion)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.09478)

+ [PointAvatar: Deformable Point-based Head Avatars from Videos](https://arxiv.org/abs/2212.08377)\
  [![Star](https://img.shields.io/github/stars/zhengyuf/pointavatar.svg?style=social&label=Star)](https://github.com/zhengyuf/pointavatar)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.08377)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://zhengyuf.github.io/PointAvatar/)

+ [PV3D: A 3D Generative Model for Portrait Video Generation](https://arxiv.org/abs/2212.06384) (ICLR 2023)\
  [![Star](https://img.shields.io/github/stars/bytedance/pv3d.svg?style=social&label=Star)](https://github.com/bytedance/pv3d)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.06384)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://showlab.github.io/pv3d/)

+ [Video Prediction by Efficient Transformers](https://arxiv.org/abs/2212.06026) (ICPR 2022)\
  [![Star](https://img.shields.io/github/stars/XiYe20/VPTR.svg?style=social&label=Star)](https://github.com/XiYe20/VPTR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.06026)

+ [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/google-research/magvit.svg?style=social&label=Star)](https://github.com/google-research/magvit)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.05199)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://magvit.cs.cmu.edu/)

+ [Physically Plausible Animation of Human Upper Body from a Single Image](https://arxiv.org/abs/2212.04741) (WACV 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.04741)

+ [MIMO Is All You Need : A Strong Multi-In-Multi-Out Baseline for Video Prediction](https://arxiv.org/pdf/2212.04655.pdf)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2212.04655.pdf)

+ [Neural Cell Video Synthesis via Optical-Flow Diffusion](https://arxiv.org/abs/2212.03250)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.03250)

+ [Video Object of Interest Segmentation](https://arxiv.org/abs/2212.02871)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.02871)

+ [Audio-Driven Co-Speech Gesture Video Generation](https://arxiv.org/abs/2212.02350) (NeurIPS 2022)\
  [![Star](https://img.shields.io/github/stars/alvinliu0/ANGIE.svg?style=social&label=Star)](https://github.com/alvinliu0/ANGIE)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.02350)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://alvinliu0.github.io/projects/ANGIE)

+ [VIDM: Video Implicit Diffusion Models](https://arxiv.org/abs/2212.00235) (AAAI 2023)\
  [![Star](https://img.shields.io/github/stars/MKFMIKU/vidm.svg?style=social&label=Star)](https://github.com/MKFMIKU/vidm)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.00235)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://kfmei.page/vidm/)

+ [Mixed Neural Voxels for Fast Multi-view Video Synthesis](https://arxiv.org/abs/2212.00190)\
  [![Star](https://img.shields.io/github/stars/fengres/mixvoxels.svg?style=social&label=Star)](https://github.com/fengres/mixvoxels)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.00190)

+ [VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Editing In the Wild](https://arxiv.org/abs/2211.14758) (SIGGRAPH Asia 2022)\
  [![Star](https://img.shields.io/github/stars/OpenTalker/video-retalking.svg?style=social&label=Star)](https://github.com/OpenTalker/video-retalking)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.14758)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://opentalker.github.io/video-retalking/)

+ [Randomized Conditional Flow Matching for Video Prediction](https://arxiv.org/abs/2211.14575)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.14575)

+ [Progressive Disentangled Representation Learning for Fine-Grained Controllable Talking Head Synthesis](https://arxiv.org/abs/2211.14506)\
  [![Star](https://img.shields.io/github/stars/Dorniwang/PD-FGC-inference.svg?style=social&label=Star)](https://github.com/Dorniwang/PD-FGC-inference)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.14506)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://dorniwang.github.io/PD-FGC/)

+ [WALDO: Future Video Synthesis using Object Layer Decomposition and Parametric Flow Prediction](https://arxiv.org/abs/2211.14308)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.14308)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://16lemoing.github.io/waldo/)

+ [Efficient Feature Extraction for High-resolution Video Frame Interpolation](https://arxiv.org/abs/2211.14005) (BMVC 2022)\
  [![Star](https://img.shields.io/github/stars/visinf/fldr-vfi.svg?style=social&label=Star)](https://github.com/visinf/fldr-vfi)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.14005)

+ [Dynamic Neural Portraits](https://arxiv.org/abs/2211.13994) (WACV 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.13994)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://michaildoukas.github.io/DynamicNeuralPortraits/)

+ [Make-A-Story: Visual Memory Conditioned Consistent Story Generation](https://arxiv.org/abs/2211.13319) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/ubc-vision/Make-A-Story.svg?style=social&label=Star)](https://github.com/ubc-vision/Make-A-Story)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.13319)

+ [Tell Me What Happened: Unifying Text-guided Video Completion via Multimodal Masked Video Generation](https://arxiv.org/abs/2211.12824) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.12824)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tvc-mmvg.github.io/)

+ [Hand Avatar: Free-Pose Hand Animation and Rendering from Monocular Video](https://arxiv.org/abs/2211.12782) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/SeanChenxy/HandAvatar.svg?style=social&label=Star)](https://github.com/SeanChenxy/HandAvatar)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.12782)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://seanchenxy.github.io/HandAvatarWeb/)

+ [SuperTran: Reference Based Video Transformer for Enhancing Low Bitrate Streams in Real Time](https://arxiv.org/abs/2211.12604#:~:text=22%20Nov%202022%5D-,SuperTran%3A%20Reference%20Based%20Video%20Transformer%20for%20Enhancing,Bitrate%20Streams%20in%20Real%20Time&text=This%20work%20focuses%20on%20low,video%20quality%20is%20severely%20compromised.)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.12604#:~:text=22%20Nov%202022%5D-,SuperTran%3A%20Reference%20Based%20Video%20Transformer%20for%20Enhancing,Bitrate%20Streams%20in%20Real%20Time&text=This%20work%20focuses%20on%20low,video%20quality%20is%20severely%20compromised.)

+ [Depth-Supervised NeRF for Multi-View RGB-D Operating Room Images](https://arxiv.org/abs/2211.12436)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.12436)

+ [SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation](https://arxiv.org/abs/2211.12194) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/OpenTalker/SadTalker.svg?style=social&label=Star)](https://github.com/OpenTalker/SadTalker)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.12194)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sadtalker.github.io/)

+ [FLEX: Full-Body Grasping Without Full-Body Grasps](https://arxiv.org/abs/2211.11903) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/purvaten/FLEX.svg?style=social&label=Star)](https://github.com/purvaten/FLEX)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.11903)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://flex.cs.columbia.edu/)

+ [Blur Interpolation Transformer for Real-World Motion from Blur](https://arxiv.org/abs/2211.11423) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/zzh-tech/BiT.svg?style=social&label=Star)](https://github.com/zzh-tech/BiT)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.11423)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://zzh-tech.github.io/BiT/)

+ [DyNCA: Real-time Dynamic Texture Synthesis Using Neural Cellular Automata](https://arxiv.org/abs/2211.11417) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/IVRL/DyNCA.svg?style=social&label=Star)](https://github.com/IVRL/DyNCA)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.11417)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://dynca.github.io/)

+ [H-VFI: Hierarchical Frame Interpolation for Videos with Large Motions](https://arxiv.org/abs/2211.11309)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.11309)

+ [AdaFNIO: Adaptive Fourier Neural Interpolation Operator for video frame interpolation](https://arxiv.org/abs/2211.10791)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.10791)

+ [SPACE: Speech-driven Portrait Animation with Controllable Expression](https://arxiv.org/abs/2211.09809)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.09809)

+ [Creative divergent synthesis with generative models](https://arxiv.org/abs/2211.08861)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.08861)

+ [CaDM: Codec-aware Diffusion Modeling for Neural-enhanced Video Streaming](https://arxiv.org/abs/2211.08428)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.08428)

+ [Advancing Learned Video Compression with In-loop Frame Prediction](https://arxiv.org/abs/2211.07004) (IEEE T-CSVT 2022)\
  [![Star](https://img.shields.io/github/stars/RenYang-home/ALVC.svg?style=social&label=Star)](https://github.com/RenYang-home/ALVC)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.07004)

+ [SSGVS: Semantic Scene Graph-to-Video Synthesis](https://arxiv.org/abs/2211.06119)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.06119)

+ [Common Pets in 3D: Dynamic New-View Synthesis of Real-Life Deformable Categories](https://arxiv.org/abs/2211.03889)\
  [![Star](https://img.shields.io/github/stars/facebookresearch/cop3d.svg?style=social&label=Star)](https://github.com/facebookresearch/cop3d)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.03889)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://cop3d.github.io/)

+ [Temporal Consistency Learning of inter-frames for Video Super-Resolution](https://arxiv.org/abs/2211.01639) (IEEE T-CSVT 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.01639)

+ [SyncTalkFace: Talking Face Generation with Precise Lip-Syncing via Audio-Lip Memory](https://arxiv.org/abs/2211.00924) (AAAI 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.00924)

+ [Learning Variational Motion Prior for Video-based Motion Capture](https://arxiv.org/abs/2210.15134)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.15134)

+ [Streaming Radiance Fields for 3D Video Synthesis](https://arxiv.org/abs/2210.14831) (NeurIPS 2022)\
  [![Star](https://img.shields.io/github/stars/AlgoHunt/StreamRF.svg?style=social&label=Star)](https://github.com/AlgoHunt/StreamRF)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.14831)

+ [Learning to forecast vegetation greenness at fine resolution over Africa with ConvLSTMs](https://arxiv.org/abs/2210.13648) (NeurIPS 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.13648)

+ [EpipolarNVS: leveraging on Epipolar geometry for single-image Novel View Synthesis](https://arxiv.org/abs/2210.13077) (BMVC 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.13077)

+ [Towards Real-Time Text2Video via CLIP-Guided, Pixel-Level Optimization](https://arxiv.org/abs/2210.12826)\
  [![Star](https://img.shields.io/github/stars/pschaldenbrand/Text2Video.svg?style=social&label=Star)](https://github.com/pschaldenbrand/Text2Video)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.12826)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://pschaldenbrand.github.io/text2video/)

+ [Facial Expression Video Generation Based-On Spatio-temporal Convolutional GAN: FEV-GAN](https://arxiv.org/abs/2210.11182) (ISWA)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.11182)

+ [Temporal and Contextual Transformer for Multi-Camera Editing of TV Shows](https://arxiv.org/abs/2210.08737) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/VirtualFilmStudio/TVMCE.svg?style=social&label=Star)](https://github.com/VirtualFilmStudio/TVMCE)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.08737)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://virtualfilmstudio.github.io/projects/multicam/)

+ [Stochastic Occupancy Grid Map Prediction in Dynamic Scenes](https://arxiv.org/abs/2210.08577)\
  [![Star](https://img.shields.io/github/stars/TempleRAIL/SOGMP.svg?style=social&label=Star)](https://github.com/TempleRAIL/SOGMP)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.08577)

+ [MonoNeRF: Learning Generalizable NeRFs from Monocular Videos without Camera Pose](https://arxiv.org/abs/2210.07181) (ICML 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.07181)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://oasisyang.github.io/mononerf/)

+ [Pre-Avatar: An Automatic Presentation Generation Framework Leveraging Talking Avatar](https://arxiv.org/abs/2210.06877) (ICTAI 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.06877)

+ [AniFaceGAN: Animatable 3D-Aware Face Image Generation for Video Avatars](https://arxiv.org/abs/2210.06465) (NeurIPS 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.06465)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yuewuhkust.github.io/AniFaceGAN/)

+ [A Generalist Framework for Panoptic Segmentation of Images and Videos](https://arxiv.org/abs/2210.06366)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.06366)

+ [Masked Motion Encoding for Self-Supervised Video Representation Learning](https://arxiv.org/abs/2210.06096) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/XinyuSun/MME.svg?style=social&label=Star)](https://github.com/XinyuSun/MME)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.06096)

+ [SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models](https://arxiv.org/abs/2210.05861) (ICLR 2023)\
  [![Star](https://img.shields.io/github/stars/pairlab/SlotFormer.svg?style=social&label=Star)](https://github.com/pairlab/SlotFormer)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.05861)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://slotformer.github.io/)

+ [Controllable Radiance Fields for Dynamic Face Synthesis](https://arxiv.org/abs/2210.05825) (3DV 2022)\
  [![Star](https://img.shields.io/github/stars/KelestZ/CoRF.svg?style=social&label=Star)](https://github.com/KelestZ/CoRF)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.05825)

+ [A unified model for continuous conditional video prediction](https://arxiv.org/abs/2210.05810) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.05810)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://npvp.github.io/)

+ [DeepHS-HDRVideo: Deep High Speed High Dynamic Range Video Reconstruction](https://arxiv.org/abs/2210.04429) (ICPR 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.04429)

+ [Self-supervised Video Representation Learning with Motion-Aware Masked Autoencoders](https://arxiv.org/abs/2210.04154)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.04154)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://github.com/happy-hsy/MotionMAE)

+ [Compressing Video Calls using Synthetic Talking Heads](https://arxiv.org/abs/2210.03692v1) (BMVC 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.03692v1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://cvit.iiit.ac.in/research/projects/cvit-projects/talking-video-compression)

+ [Text-driven Video Prediction](https://arxiv.org/abs/2210.02872)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02872)

+ [Audio-Visual Face Reenactment](https://arxiv.org/abs/2210.02755) (WACV 2023)\
  [![Star](https://img.shields.io/github/stars/mdv3101/AVFR-Gan.svg?style=social&label=Star)](https://github.com/mdv3101/AVFR-Gan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02755)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://cvit.iiit.ac.in/research/projects/cvit-projects/avfr)

+ [Geometry Driven Progressive Warping for One-Shot Face Animation](https://arxiv.org/abs/2210.02391)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02391)

+ [Cross-identity Video Motion Retargeting with Joint Transformation and Synthesis](https://arxiv.org/abs/2210.01559) (WACV 2023)\
  [![Star](https://img.shields.io/github/stars/nihaomiao/WACV23_TSNet.svg?style=social&label=Star)](https://github.com/nihaomiao/WACV23_TSNet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.01559)

+ [Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset](https://arxiv.org/abs/2209.12475) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/zmzhang1998/Real-RawVSR.svg?style=social&label=Star)](https://github.com/zmzhang1998/Real-RawVSR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.12475)

+ [VToonify: Controllable High-Resolution Portrait Video Style Transfer](https://arxiv.org/abs/2209.11224) (SIGGRAPH Asia 2022)\
  [![Star](https://img.shields.io/github/stars/williamyang1991/VToonify.svg?style=social&label=Star)](https://github.com/williamyang1991/VToonify)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.11224)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.mmlab-ntu.com/project/vtoonify/)

+ [T3VIP: Transformation-based 3D Video Prediction](https://arxiv.org/abs/2209.11693) (IEEE)\
  [![Star](https://img.shields.io/github/stars/nematoli/t3vip.svg?style=social&label=Star)](https://github.com/nematoli/t3vip)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.11693)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://t3vip.cs.uni-freiburg.de/)

+ [NeuralMarker: A Framework for Learning General Marker Correspondence](https://arxiv.org/abs/2209.08896) (SIGGRAPH Asia 2022)\
  [![Star](https://img.shields.io/github/stars/drinkingcoder/NeuralMarker.svg?style=social&label=Star)](https://github.com/drinkingcoder/NeuralMarker)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.08896)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://drinkingcoder.github.io/publication/neuralmarker/)

+ [AutoLV: Automatic Lecture Video Generator](https://arxiv.org/abs/2209.08795)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.08795)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.youtube.com/watch?v=cY6TYkI0cog)

+ [Continuously Controllable Facial Expression Editing in Talking Face Videos](https://arxiv.org/abs/2209.08289)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.08289)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.youtube.com/watch?v=WD-bNVya6kM)

+ [A Deep Moving-camera Background Model](https://arxiv.org/abs/2209.07923) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/BGU-CS-VIL/DeepMCBM.svg?style=social&label=Star)](https://github.com/BGU-CS-VIL/DeepMCBM)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.07923)

+ [HARP: Autoregressive Latent Video Prediction with High-Fidelity Image Generator](https://arxiv.org/abs/2209.07143) (ICIP 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.07143)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/harp-videos/home)

+ [Talking Head from Speech Audio using a Pre-trained Image Generator](https://arxiv.org/abs/2209.04252) (ACM Multimedia 2022)\
  [![Star](https://img.shields.io/github/stars/MohammedAlghamdi/talking-heads-acm-mm.svg?style=social&label=Star)](https://github.com/MohammedAlghamdi/talking-heads-acm-mm)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.04252)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://mohammedalghamdi.github.io/talking-heads-acm-mm/)

+ [Treating Motion as Option to Reduce Motion Dependency in Unsupervised Video Object Segmentation](https://arxiv.org/abs/2209.03138) (WACV 2023)\
  [![Star](https://img.shields.io/github/stars/suhwan-cho/tmo.svg?style=social&label=Star)](https://github.com/suhwan-cho/tmo)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.03138)

+ [Neural Sign Reenactor: Deep Photorealistic Sign Language Retargeting](https://arxiv.org/abs/2209.01470) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.01470)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.youtube.com/watch?v=xKAfguacOkE)

+ [REMOT: A Region-to-Whole Framework for Realistic Human Motion Transfer](https://arxiv.org/abs/2209.00475) (ACMMM 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.00475)

+ [SketchBetween: Video-to-Video Synthesis for Sprite Animation via Sketches](https://arxiv.org/abs/2209.00185) (ACM conference on the Foundations of Digital Games)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.00185)

+ [StableFace: Analyzing and Improving Motion Stability for Talking Face Generation](https://arxiv.org/abs/2208.13717)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.13717)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://stable-face.github.io/)

+ [VMFormer: End-to-End Video Matting with Transformer](https://arxiv.org/abs/2208.12801)\
  [![Star](https://img.shields.io/github/stars/SHI-Labs/VMFormer.svg?style=social&label=Star)](https://github.com/SHI-Labs/VMFormer)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.12801)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://chrisjuniorli.github.io/project/VMFormer/)

+ [Neural Novel Actor: Learning a Generalized Animatable Neural Representation for Human Actors](https://arxiv.org/abs/2208.11905)\
  [![Star](https://img.shields.io/github/stars/Talegqz/neural_novel_actor.svg?style=social&label=Star)](https://github.com/Talegqz/neural_novel_actor)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.11905)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://talegqz.github.io/neural_novel_actor/)

+ [StyleTalker: One-shot Style-based Audio-driven Talking Head Video Generation](https://arxiv.org/abs/2208.10922)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.10922)

+ [Towards MOOCs for Lipreading: Using Synthetic Talking Heads to Train Humans in Lipreading at Scale](https://arxiv.org/abs/2208.09796) (WACV 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.09796)

+ [Temporal View Synthesis of Dynamic Scenes through 3D Object Motion Estimation with Multi-Plane Images](https://arxiv.org/abs/2208.09463) (ISMAR 2022)\
  [![Star](https://img.shields.io/github/stars/NagabhushanSN95/DeCOMPnet.svg?style=social&label=Star)](https://github.com/NagabhushanSN95/DeCOMPnet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.09463)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://nagabhushansn95.github.io/publications/2022/DeCOMPnet.html)

+ [Wildfire Forecasting with Satellite Images and Deep Generative Model](https://arxiv.org/abs/2208.09411)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.09411)

+ [Video Interpolation by Event-driven Anisotropic Adjustment of Optical Flow](https://arxiv.org/abs/2208.09127) (ECCV 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.09127)

+ [Extreme-scale Talking-Face Video Upsampling with Audio-Visual Priors](https://arxiv.org/abs/2208.08118) (ACMMM 2022)\
  [![Star](https://img.shields.io/github/stars/Sindhu-Hegde/video-super-resolver.svg?style=social&label=Star)](https://github.com/Sindhu-Hegde/video-super-resolver)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.08118)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://cvit.iiit.ac.in/research/projects/cvit-projects/talking-face-video-upsampling)

+ [Semi-Supervised Video Inpainting with Cycle Consistency Constraints](https://arxiv.org/abs/2208.06807)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.06807)

+ [UAV-CROWD: Violent and non-violent crowd activity simulator from the perspective of UAV](https://arxiv.org/abs/2208.06702)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.06702)

+ [Cine-AI: Generating Video Game Cutscenes in the Style of Human Directors](https://arxiv.org/abs/2208.05701) (ACMHCI)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.05701)

+ [Language-Guided Face Animation by Recurrent StyleGAN-based Generator](https://arxiv.org/abs/2208.05617)\
  [![Star](https://img.shields.io/github/stars/researchmm/language-guided-animation.svg?style=social&label=Star)](https://github.com/researchmm/language-guided-animation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.05617)

+ [Boosting neural video codecs by exploiting hierarchical redundancy](https://arxiv.org/abs/2208.04303)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.04303)

+ [PS-NeRV: Patch-wise Stylized Neural Representations for Videos](https://arxiv.org/abs/2208.03742)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.03742)

+ [Real-time Gesture Animation Generation from Speech for Virtual Human Interaction](https://arxiv.org/abs/2208.03244) (CHI EA 2021)\
  [![Star](https://img.shields.io/github/stars/mrebol/Gestures-From-Speech.svg?style=social&label=Star)](https://github.com/mrebol/Gestures-From-Speech)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.03244)

+ [Meta-Interpolation: Time-Arbitrary Frame Interpolation via Dual Meta-Learning](https://arxiv.org/abs/2207.13670)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.13670)

+ [Efficient Video Deblurring Guided by Motion Magnitude](https://arxiv.org/abs/2207.13374) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/sollynoay/MMP-RNN.svg?style=social&label=Star)](https://github.com/sollynoay/MMP-RNN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.13374)

+ [Error-Aware Spatial Ensembles for Video Frame Interpolation](https://arxiv.org/abs/2207.12305)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.12305)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.youtube.com/watch?v=_32GNANSr5U)

+ [Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis](https://arxiv.org/abs/2207.11770) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/sstzal/DFRF.svg?style=social&label=Star)](https://github.com/sstzal/DFRF)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.11770)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sstzal.github.io/DFRF/)

+ [InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images](https://arxiv.org/abs/2207.11148) (ECCV 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.11148)

+ [RealFlow: EM-based Realistic Optical Flow Dataset Generation from Videos](https://arxiv.org/abs/2207.11075) (ECCV 2022 Oral)\
  [![Star](https://img.shields.io/github/stars/megvii-research/RealFlow.svg?style=social&label=Star)](https://github.com/megvii-research/RealFlow)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.11075)

+ [Towards Interpretable Video Super-Resolution via Alternating Optimization](https://arxiv.org/abs/2207.10765) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/caojiezhang/DAVSR.svg?style=social&label=Star)](https://github.com/caojiezhang/DAVSR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.10765)

+ [Error Compensation Framework for Flow-Guided Video Inpainting](https://arxiv.org/abs/2207.10391) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/JaeYeonKang/ECFVI.svg?style=social&label=Star)](https://github.com/JaeYeonKang/ECFVI)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.10391)

+ [Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance](https://arxiv.org/abs/2207.10123) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/zzh-tech/Animation-from-Blur.svg?style=social&label=Star)](https://github.com/zzh-tech/Animation-from-Blur)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.10123)

+ [TTVFI: Learning Trajectory-Aware Transformer for Video Frame Interpolation](https://arxiv.org/abs/2207.09048) (CVPR 2022 Oral)\
  [![Star](https://img.shields.io/github/stars/researchmm/TTVSR.svg?style=social&label=Star)](https://github.com/researchmm/TTVSR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.09048)

+ [Audio Input Generates Continuous Frames to Synthesize Facial Video Using Generative Adiversarial Networks](https://arxiv.org/abs/2207.08813)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.08813)

+ [Neighbor Correspondence Matching for Flow-based Video Frame Synthesis](https://arxiv.org/abs/2207.06763) (ACMMM 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.06763)

+ [You Only Align Once: Bidirectional Interaction for Spatial-Temporal Video Super-Resolution](https://arxiv.org/abs/2207.06345) (ACMMM 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.06345)

+ [CANF-VC: Conditional Augmented Normalizing Flows for Video Compression](https://arxiv.org/abs/2207.05315)\
  [![Star](https://img.shields.io/github/stars/NYCU-MAPL/CANF-VC.svg?style=social&label=Star)](https://github.com/NYCU-MAPL/CANF-VC)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.05315)

+ [A Probabilistic Model Of Interaction Dynamics for Dyadic Face-to-Face Settings](https://arxiv.org/abs/2207.04566)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.04566)

+ [Cross-Attention Transformer for Video Interpolation](https://arxiv.org/abs/2207.04132)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.04132)

+ [Jointly Harnessing Prior Structures and Temporal Consistency for Sign Language Video Generation](https://arxiv.org/abs/2207.03714)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.03714)

+ [Segmenting Moving Objects via an Object-Centric Layered Representation](https://arxiv.org/abs/2207.02206) (NeurIPS 2022)\
  [![Star](https://img.shields.io/github/stars/jyxarthur/oclr_model.svg?style=social&label=Star)](https://github.com/jyxarthur/oclr_model)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.02206)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.robots.ox.ac.uk/~vgg/research/oclr/)

+ [Programmatic Concept Learning for Human Motion Description and Synthesis](https://arxiv.org/abs/2206.13502) (CVPR 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.13502)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sumith1896.github.io/motion-concepts/)

+ [Optimizing Video Prediction via Video Frame Interpolation](https://arxiv.org/abs/2206.13454) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/YueWuHKUST/CVPR2022-Optimizing-Video-Prediction-via-Video-Frame-Interpolation.svg?style=social&label=Star)](https://github.com/YueWuHKUST/CVPR2022-Optimizing-Video-Prediction-via-Video-Frame-Interpolation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.13454)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yuewuhkust.github.io/OVP_VFI/)

+ [Perceptual Conversational Head Generation with Regularized Driver and Enhanced Renderer](https://arxiv.org/abs/2206.12837) (ACMMM 2022)\
  [![Star](https://img.shields.io/github/stars/megvii-research/MM2022-ViCoPerceptualHeadGeneration.svg?style=social&label=Star)](https://github.com/megvii-research/MM2022-ViCoPerceptualHeadGeneration)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.12837)

+ [MaskViT: Masked Visual Pre-Training for Video Prediction](https://arxiv.org/abs/2206.11894)\
  [![Star](https://img.shields.io/github/stars/agrimgupta92/maskvit.svg?style=social&label=Star)](https://github.com/agrimgupta92/maskvit)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.11894)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://maskedvit.github.io/)

+ [Enhanced Bi-directional Motion Estimation for Video Frame Interpolation](https://arxiv.org/abs/2206.08572) (WACV 2023)\
  [![Star](https://img.shields.io/github/stars/srcn-ivl/EBME.svg?style=social&label=Star)](https://github.com/srcn-ivl/EBME)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.08572)

+ [Face-Dubbing++: Lip-Synchronous, Voice Preserving Translation of Videos](https://arxiv.org/abs/2206.04523)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.04523)

+ [STIP: A SpatioTemporal Information-Preserving and Perception-Augmented Model for High-Resolution Video Prediction](https://arxiv.org/abs/2206.04381) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/ZhengChang467/STIPHR.svg?style=social&label=Star)](https://github.com/ZhengChang467/STIPHR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.04381)

+ [JNMR: Joint Non-linear Motion Regression for Video Frame Interpolation](https://arxiv.org/abs/2206.04231v2)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.04231v2)

+ [SimVP: Simpler yet Better Video Prediction](https://arxiv.org/abs/2206.05099) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/A4Bio/SimVP-Simpler-yet-Better-Video-Prediction.svg?style=social&label=Star)](https://github.com/A4Bio/SimVP-Simpler-yet-Better-Video-Prediction)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.05099)

+ [Recurrent Video Restoration Transformer with Guided Deformable Attention](https://arxiv.org/abs/2206.02146) (NeurIPS 2022)\
  [![Star](https://img.shields.io/github/stars/JingyunLiang/RVRT.svg?style=social&label=Star)](https://github.com/JingyunLiang/RVRT)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.02146)

+ [Cascaded Video Generation for Videos In-the-Wild](https://arxiv.org/abs/2206.00735) (ICPR 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.00735)

+ [D$^2$NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video](https://arxiv.org/abs/2205.15838)\
  [![Star](https://img.shields.io/github/stars/ChikaYan/d2nerf.svg?style=social&label=Star)](https://github.com/ChikaYan/d2nerf)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.15838)

+ [TubeFormer-DeepLab: Video Mask Transformer](https://arxiv.org/abs/2205.15361) (CVPR 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.15361)

+ [IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation](https://arxiv.org/abs/2205.14620) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/ltkong218/IFRNet.svg?style=social&label=Star)](https://github.com/ltkong218/IFRNet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.14620)

+ [Feature-Aligned Video Raindrop Removal with Temporal Constraints](https://arxiv.org/abs/2205.14574)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.14574)

+ [Future Transformer for Long-term Action Anticipation](https://arxiv.org/abs/2205.14022) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/gongda0e/FUTR.svg?style=social&label=Star)](https://github.com/gongda0e/FUTR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.14022)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://cvlab.postech.ac.kr/research/FUTR/)

+ [Video2StyleGAN: Disentangling Local and Global Variations in a Video](https://arxiv.org/abs/2205.13996)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.13996)

+ [Automatic Generation of Synthetic Colonoscopy Videos for Domain Randomization](https://arxiv.org/abs/2205.10368)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.10368)

+ [Latent-space disentanglement with untrained generator networks for the isolation of different motion types in video data](https://arxiv.org/abs/2205.10367)\
  [![Star](https://img.shields.io/github/stars/hollerm/generator_based_motion_isolation.svg?style=social&label=Star)](https://github.com/hollerm/generator_based_motion_isolation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.10367)

+ [Video Frame Interpolation with Transformer](https://arxiv.org/abs/2205.07230) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/dvlab-research/VFIformer.svg?style=social&label=Star)](https://github.com/dvlab-research/VFIformer)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.07230)

+ [Multi-encoder Network for Parameter Reduction of a Kernel-based Interpolation Architecture](https://arxiv.org/abs/2205.06723) (NTIRE)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.06723)

+ [Diverse Video Generation from a Single Video](https://arxiv.org/abs/2205.05725) (CVPR 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.05725)

+ [Video-ReTime: Learning Temporally Varying Speediness for Time Remapping](https://arxiv.org/abs/2205.05609) (AICC)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.05609)

+ [Spatial-Temporal Space Hand-in-Hand: Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning](https://arxiv.org/abs/2205.05264)\
  [![Star](https://img.shields.io/github/stars/hhhhhumengshun/SFI-STVR.svg?style=social&label=Star)](https://github.com/hhhhhumengshun/SFI-STVR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.05264)

+ [Image2Gif: Generating Continuous Realistic Animations with Warping NODEs](https://arxiv.org/abs/2205.04519) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/JurijsNazarovs/warping_node.svg?style=social&label=Star)](https://github.com/JurijsNazarovs/warping_node)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.04519)

+ [GAN-Based Multi-View Video Coding with Spatio-Temporal EPI Reconstruction](https://arxiv.org/abs/2205.03599)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.03599)

+ [Parametric Reshaping of Portraits in Videos](https://arxiv.org/abs/2205.02538v1)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.02538v1)

+ [Video Extrapolation in Space and Time](https://arxiv.org/abs/2205.02084) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/zzyunzhi/vest.svg?style=social&label=Star)](https://github.com/zzyunzhi/vest)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.02084)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://cs.stanford.edu/~yzzhang/projects/vest/)

+ [Zero-Episode Few-Shot Contrastive Predictive Coding: Solving intelligence tests without prior training](https://arxiv.org/abs/2205.01924)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.01924)

+ [Copy Motion From One to Another: Fake Motion Video Generation](https://arxiv.org/abs/2205.01373)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.01373)

+ [Neural Implicit Representations for Physical Parameter Inference from a Single Video](https://arxiv.org/abs/2204.14030) (WACV 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.14030)

+ [Talking Head Generation Driven by Speech-Related Facial Action Units and Audio- Based on Multimodal Representation Fusion](https://arxiv.org/abs/2110.09951) (BMVC 2021)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.09951)

+ [ClothFormer:Taming Video Virtual Try-on in All Module](https://arxiv.org/abs/2204.12151) (CVPR 2022 Oral)\
  [![Star](https://img.shields.io/github/stars/luxiangju-PersonAI/ClothFormer.svg?style=social&label=Star)](https://github.com/luxiangju-PersonAI/ClothFormer)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.12151)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://cloth-former.github.io/)

+ [Future Object Detection with Spatiotemporal Transformers](https://arxiv.org/abs/2204.10321)\
  [![Star](https://img.shields.io/github/stars/atonderski/future-object-detection.svg?style=social&label=Star)](https://github.com/atonderski/future-object-detection)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.10321)

+ [STAU: A SpatioTemporal-Aware Unit for Video Prediction and Beyond](https://arxiv.org/abs/2204.09456) (TPAMI)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.09456)

+ [Sound-Guided Semantic Video Generation](https://arxiv.org/abs/2204.09273) (ECCV 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.09273)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://kuai-lab.github.io/eccv2022sound/)

+ [Less than Few: Self-Shot Video Instance Segmentation](https://arxiv.org/abs/2204.08874)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.08874)

+ [Learning to Listen: Modeling Non-Deterministic Dyadic Facial Motion](https://arxiv.org/abs/2204.08451)\
  [![Star](https://img.shields.io/github/stars/evonneng/learning2listen.svg?style=social&label=Star)](https://github.com/evonneng/learning2listen)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.08451)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://evonneng.github.io/learning2listen/)

+ [MUGEN: A Playground for Video-Audio-Text Multimodal Understanding and GENeration](https://arxiv.org/abs/2204.08058)\
  [![Star](https://img.shields.io/github/stars/mugen-org/MUGEN_baseline.svg?style=social&label=Star)](https://github.com/mugen-org/MUGEN_baseline)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.08058)

+ [Controllable Video Generation through Global and Local Motion Dynamics](https://arxiv.org/abs/2204.06558)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.06558)

+ [Dynamic Neural Textures: Generating Talking-Face Videos with Continuously Controllable Expressions](https://arxiv.org/abs/2204.06180)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.06180)

+ [Self-Supervised Traffic Advisors: Distributed, Multi-view Traffic Prediction for Smart Cities](https://arxiv.org/abs/2204.06171) (ITSC)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.06171)

+ [Structure-Aware Motion Transfer with Deformable Anchor Model](https://arxiv.org/abs/2204.05018) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/JialeTao/DAM.svg?style=social&label=Star)](https://github.com/JialeTao/DAM)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.05018)

+ [HSTR-Net: High Spatio-Temporal Resolution Video Generation For Wide Area Surveillance](https://arxiv.org/abs/2204.04435)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.04435)

+ [SunStage: Portrait Reconstruction and Relighting using the Sun as a Light Stage](https://arxiv.org/abs/2204.03648) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/adobe-research/sunstage.svg?style=social&label=Star)](https://github.com/adobe-research/sunstage)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.03648)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sunstage.cs.washington.edu/)

+ [Many-to-many Splatting for Efficient Video Frame Interpolation](https://arxiv.org/abs/2204.03513) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/feinanshan/M2M_VFI.svg?style=social&label=Star)](https://github.com/feinanshan/M2M_VFI)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.03513)

+ [Video Demoireing with Relation-Based Temporal Consistency](https://arxiv.org/abs/2204.02957) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/CVMI-Lab/VideoDemoireing.svg?style=social&label=Star)](https://github.com/CVMI-Lab/VideoDemoireing)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.02957)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://daipengwa.github.io/VDmoire_ProjectPage/)

+ [Neural Rendering of Humans in Novel View and Pose from Monocular Video](https://arxiv.org/abs/2204.01218)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.01218)

+ [MPS-NeRF: Generalizable 3D Human Rendering from Multiview Images](https://arxiv.org/abs/2203.16875) (TPAMI 2022)\
  [![Star](https://img.shields.io/github/stars/gaoxiangjun/MPS-NeRF.svg?style=social&label=Star)](https://github.com/gaoxiangjun/MPS-NeRF)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.16875)

+ [Foveation-based Deep Video Compression without Motion Search](https://arxiv.org/abs/2203.16490)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.16490)

+ [STRPM: A Spatiotemporal Residual Predictive Model for High-Resolution Video Prediction](https://arxiv.org/abs/2203.16084) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/ZhengChang467/STIPHR.svg?style=social&label=Star)](https://github.com/ZhengChang467/STIPHR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.16084)

+ [High-resolution Face Swapping via Latent Semantics Disentanglement](https://arxiv.org/abs/2203.15958) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/cnnlstm/FSLSD_HiRes.svg?style=social&label=Star)](https://github.com/cnnlstm/FSLSD_HiRes)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.15958)

+ [VPTR: Efficient Transformers for Video Prediction](https://arxiv.org/abs/2203.15836) (ICPR 2022)\
  [![Star](https://img.shields.io/github/stars/XiYe20/VPTR.svg?style=social&label=Star)](https://github.com/XiYe20/VPTR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.15836)

+ [Long-term Video Frame Interpolation via Feature Propagation](https://arxiv.org/abs/2203.15427) (CVPR 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.15427)

+ [Signing at Scale: Learning to Co-Articulate Signs for Large-Scale Photo-Realistic Sign Language Production](https://arxiv.org/abs/2203.15354)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.15354)

+ [Dressing in the Wild by Watching Dance Videos](https://arxiv.org/abs/2203.15320) (CVPR 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.15320)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://awesome-wflow.github.io/)

+ [Structured Local Radiance Fields for Human Avatar Modeling](https://arxiv.org/abs/2203.14478) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/ZhengZerong/THUman4.0-Dataset.svg?style=social&label=Star)](https://github.com/ZhengZerong/THUman4.0-Dataset)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.14478)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://www.liuyebin.com/slrf/slrf.html)

+ [V3GAN: Decomposing Background, Foreground and Motion for Video Generation](https://arxiv.org/abs/2203.14074)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.14074)

+ [Keypoints Tracking via Transformer Networks](https://arxiv.org/abs/2203.12848)\
  [![Star](https://img.shields.io/github/stars/LexaNagiBator228/Keypoints-Tracking-via-Transformer-Networks.svg?style=social&label=Star)](https://github.com/LexaNagiBator228/Keypoints-Tracking-via-Transformer-Networks)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.12848)

+ [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) (NeurIPS 2022)\
  [![Star](https://img.shields.io/github/stars/MCG-NJU/VideoMAE.svg?style=social&label=Star)](https://github.com/MCG-NJU/VideoMAE)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.12602)

+ [Unifying Motion Deblurring and Frame Interpolation with Events](https://arxiv.org/abs/2203.12178) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/XiangZ-0/EVDI.svg?style=social&label=Star)](https://github.com/XiangZ-0/EVDI)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.12178)

+ [QS-Craft: Learning to Quantize, Scrabble and Craft for Conditional Human Motion Animation](https://arxiv.org/abs/2203.11632)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.11632)

+ [Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields](https://arxiv.org/abs/2203.10821) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/donydchen/sem2nerf.svg?style=social&label=Star)](https://github.com/donydchen/sem2nerf)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.10821)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://donydchen.github.io/sem2nerf/)

+ [Stochastic Video Prediction with Structure and Motion](https://arxiv.org/abs/2203.10528) (TPAMI)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.10528)

+ [Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation](https://arxiv.org/abs/2203.10291)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.10291)

+ [Beyond a Video Frame Interpolator: A Space Decoupled Learning Approach to Continuous Image Transition](https://arxiv.org/abs/2203.09771)\
  [![Star](https://img.shields.io/github/stars/yangxy/SDL.svg?style=social&label=Star)](https://github.com/yangxy/SDL)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09771)

+ [Transframer: Arbitrary Frame Prediction with Generative Models](https://arxiv.org/abs/2203.09494)\
  [![Star](https://img.shields.io/github/stars/lucidrains/transframer-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/transframer-pytorch)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09494)

+ [Look Outside the Room: Synthesizing A Consistent Long-Term 3D Scene Video from A Single Image](https://arxiv.org/abs/2203.09457) (CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/xrenaa/Look-Outside-Room.svg?style=social&label=Star)](https://github.com/xrenaa/Look-Outside-Room)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09457)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://xrenaa.github.io/look-outside-room/)

+ [MSPred: Video Prediction at Multiple Spatio-Temporal Scales with Hierarchical Recurrent Networks](https://arxiv.org/abs/2203.09303)\
  [![Star](https://img.shields.io/github/stars/AIS-Bonn/MSPred.svg?style=social&label=Star)](https://github.com/AIS-Bonn/MSPred)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09303)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/mspred/home)

+ [Latent Image Animator: Learning to Animate Images via Latent Space Navigation](https://arxiv.org/abs/2203.09043) (ICLR 2022)\
  [![Star](https://img.shields.io/github/stars/wyhsirius/LIA.svg?style=social&label=Star)](https://github.com/wyhsirius/LIA)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09043)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://wyhsirius.github.io/LIA-project/)

+ [DialogueNeRF: Towards Realistic Avatar Face-to-face Conversation Video Generation](https://arxiv.org/abs/2203.07931)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.07931)

+ [One-stage Video Instance Segmentation: From Frame-in Frame-out to Clip-in Clip-out](https://arxiv.org/abs/2203.06421)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.06421)

+ [NeRFocus: Neural Radiance Field for 3D Synthetic Defocus](https://arxiv.org/abs/2203.05189)\
  [![Star](https://img.shields.io/github/stars/wyhuai/NeRFocus.svg?style=social&label=Star)](https://github.com/wyhuai/NeRFocus)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.05189)

+ [A Novel Dual Dense Connection Network for Video Super-resolution](https://arxiv.org/abs/2203.02723)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.02723)

+ [Region-of-Interest Based Neural Video Compression](https://arxiv.org/abs/2203.01978) (BMVC 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.01978)

+ [Thinking the Fusion Strategy of Multi-reference Face Reenactment](https://arxiv.org/abs/2202.10758v1) (ICIP 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.10758v1)

+ [Neural Marionette: Unsupervised Learning of Motion Skeleton and Latent Dynamics from Volumetric Video](https://arxiv.org/abs/2202.08418) (AAAI 2022)\
  [![Star](https://img.shields.io/github/stars/jinseokbae/neural_marionette.svg?style=social&label=Star)](https://github.com/jinseokbae/neural_marionette)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.08418)

+ [Enhancing Deformable Convolution based Video Frame Interpolation with Coarse-to-fine 3D CNN](https://arxiv.org/abs/2202.07731)\
  [![Star](https://img.shields.io/github/stars/danier97/EDC.svg?style=social&label=Star)](https://github.com/danier97/EDC)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.07731)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://danier97.github.io/EDC/)

+ [Exploring Discontinuity for Video Frame Interpolation](https://arxiv.org/abs/2202.07291) (CVPR 2023)\
  [![Star](https://img.shields.io/github/stars/pandatimo/Exploring-Discontinuity-for-VFI.svg?style=social&label=Star)](https://github.com/pandatimo/Exploring-Discontinuity-for-VFI)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.07291)

+ [A new face swap method for image and video domains: a technical report](https://arxiv.org/abs/2202.03046v1)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.03046v1)

+ [Feature-Style Encoder for Style-Based GAN Inversion](https://arxiv.org/abs/2202.02183)\
  [![Star](https://img.shields.io/github/stars/InterDigitalInc/FeatureStyleEncoder.svg?style=social&label=Star)](https://github.com/InterDigitalInc/FeatureStyleEncoder)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.02183)

+ [Third Time's the Charm? Image and Video Editing with StyleGAN3](https://arxiv.org/abs/2201.13433)\
  [![Star](https://img.shields.io/github/stars/yuval-alaluf/stylegan3-editing.svg?style=social&label=Star)](https://github.com/yuval-alaluf/stylegan3-editing)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.13433)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yuval-alaluf.github.io/stylegan3-editing/)

+ [Deep Video Prior for Video Consistency and Propagation](https://arxiv.org/abs/2201.11632) (TPAMI 2021)\
  [![Star](https://img.shields.io/github/stars/ChenyangLEI/deep-video-prior.svg?style=social&label=Star)](https://github.com/ChenyangLEI/deep-video-prior)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.11632)

+ [Non-linear Motion Estimation for Video Frame Interpolation using Space-time Convolutions](https://arxiv.org/abs/2201.11407) (CLIC, CVPR 2022)\
  [![Star](https://img.shields.io/github/stars/saikatdutta/NME-VFI.svg?style=social&label=Star)](https://github.com/saikatdutta/NME-VFI)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.11407)

+ [Splatting-based Synthesis for Video Frame Interpolation](https://arxiv.org/abs/2201.10075) (WACV 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.10075)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://sniklaus.com/splatsyn)

+ [Stitch it in Time: GAN-Based Facial Editing of Real Videos](https://arxiv.org/abs/2201.08361)\
  [![Star](https://img.shields.io/github/stars/rotemtzaban/STIT.svg?style=social&label=Star)](https://github.com/rotemtzaban/STIT)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.08361)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://stitch-time.github.io/)

+ [Self-Supervised Deep Blind Video Super-Resolution](https://arxiv.org/abs/2201.07422)\
  [![Star](https://img.shields.io/github/stars/csbhr/Self-Blind-VSR.svg?style=social&label=Star)](https://github.com/csbhr/Self-Blind-VSR)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.07422)

+ [Autoencoding Video Latents for Adversarial Video Generation](https://arxiv.org/abs/2201.06888)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.06888)

+ [AugLy: Data Augmentations for Robustness](https://arxiv.org/abs/2201.06494)\
  [![Star](https://img.shields.io/github/stars/facebookresearch/AugLy.svg?style=social&label=Star)](https://github.com/facebookresearch/AugLy)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.06494)

+ [Towards Realistic Visual Dubbing with Heterogeneous Sources](https://arxiv.org/abs/2201.06260) (ACMMM 2021)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.06260)

+ [Audio-Driven Talking Face Video Generation with Dynamic Convolution Kernels](https://arxiv.org/abs/2201.05986) (IEEE)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.05986)

+ [Learning Temporally and Semantically Consistent Unpaired Video-to-video Translation Through Pseudo-Supervision From Synthetic Optical Flow](https://arxiv.org/abs/2201.05723) (AAAI 2022)\
  [![Star](https://img.shields.io/github/stars/Unsup_Recycle_GAN/.svg?style=social&label=Star)](https://github.com/wangkaihong/Unsup_Recycle_GAN/)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.05723)

+ [MetaDance: Few-shot Dancing Video Retargeting via Temporal-aware Meta-learning](https://arxiv.org/abs/2201.04851)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.04851)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://github.com/geyuying/MetaDance)

+ [Music2Video: Automatic Generation of Music Video with fusion of audio and text](https://arxiv.org/abs/2201.03809)\
  [![Star](https://img.shields.io/github/stars/joeljang/music2video.svg?style=social&label=Star)](https://github.com/joeljang/music2video)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.03809)

+ [MobileFaceSwap: A Lightweight Framework for Video Face Swapping](https://arxiv.org/abs/2201.03808) (AAAI 2022)\
  [![Star](https://img.shields.io/github/stars/Seanseattle/MobileFaceSwap.svg?style=social&label=Star)](https://github.com/Seanseattle/MobileFaceSwap)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.03808)

+ [Structured 3D Features for Reconstructing Controllable Avatars](https://arxiv.org/abs/2212.06820) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.06820)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://enriccorona.github.io/s3f/)

+ [MoFusion: A Framework for Denoising-Diffusion-based Motion Synthesis](https://arxiv.org/abs/2212.04495) (CVPR 2023)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.04495)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vcai.mpi-inf.mpg.de/projects/MoFusion/)

+ [High-fidelity Facial Avatar Reconstruction from Monocular Video with Generative Priors](https://arxiv.org/abs/2211.15064)\
  [![Star](https://img.shields.io/github/stars/bbaaii/HFA-GP.svg?style=social&label=Star)](https://github.com/bbaaii/HFA-GP)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.15064)

+ [3DDesigner: Towards Photorealistic 3D Object Generation and Editing with Text-guided Diffusion Models](https://arxiv.org/abs/2211.14108)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.14108)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://3ddesigner-diffusion.github.io/)

+ [Audio-visual video face hallucination with frequency supervision and cross modality support by speech based lip reading loss](https://arxiv.org/abs/2211.10883)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.10883)

+ [It Takes Two: Masked Appearance-Motion Modeling for Self-supervised Video Transformer Pre-training](https://arxiv.org/abs/2210.05234)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.05234)

+ [See, Plan, Predict: Language-guided Cognitive Planning with Video Prediction](https://arxiv.org/abs/2210.03825)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.03825)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://see-pp.github.io/)

+ [Motion Transformer for Unsupervised Image Animation](https://arxiv.org/abs/2209.14024) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/JialeTao/MoTrans.svg?style=social&label=Star)](https://github.com/JialeTao/MoTrans)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.14024)

+ [Low-Light Video Enhancement with Synthetic Event Guidance](https://arxiv.org/abs/2208.11014)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.11014)

+ [Neural Capture of Animatable 3D Human from Monocular Video](https://arxiv.org/abs/2208.08728) (ECCV 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.08728)

+ [NDF: Neural Deformable Fields for Dynamic Human Modelling](https://arxiv.org/abs/2207.09193) (ECCV 2022)\
  [![Star](https://img.shields.io/github/stars/HKBU-VSComputing/2022_ECCV_NDF.svg?style=social&label=Star)](https://github.com/HKBU-VSComputing/2022_ECCV_NDF)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.09193)

+ [Diverse Dance Synthesis via Keyframes with Transformer Controllers](https://arxiv.org/abs/2207.05906)\
  [![Star](https://img.shields.io/github/stars/godzillalla/Dance-Synthesis-Project.svg?style=social&label=Star)](https://github.com/godzillalla/Dance-Synthesis-Project)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2207.05906)

+ [CTrGAN: Cycle Transformers GAN for Gait Transfer](https://arxiv.org/abs/2206.15248)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.15248)

+ [Enhanced Deep Animation Video Interpolation](https://arxiv.org/abs/2206.12657)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.12657)

+ [An Identity-Preserved Framework for Human Motion Transfer](https://arxiv.org/abs/2204.06862)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.06862)

+ [Unsupervised Coherent Video Cartoonization with Perceptual Motion Consistency](https://arxiv.org/abs/2204.00795)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.00795)

+ [Learning Multi-Object Dynamics with Compositional Neural Radiance Fields](https://arxiv.org/abs/2202.11855) (CoRL 2022)\
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.11855)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://dannydriess.github.io/compnerfdyn/)

+ [VRT: A Video Restoration Transformer](https://arxiv.org/abs/2201.12288)\
  [![Star](https://img.shields.io/github/stars/JingyunLiang/VRT.svg?style=social&label=Star)](https://github.com/JingyunLiang/VRT)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2201.12288)


<br>


## 2021

+ [NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion](https://arxiv.org/pdf/2111.12417)  
  [![Star](https://img.shields.io/github/stars/lucidrains/nuwa-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/nuwa-pytorch)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.12417)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.microsoft.com/en-us/research/project/nuwa-infinity/)

+ [Generative Adversarial Graph Convolutional Networks for Human Action Synthesis](https://arxiv.org/pdf/2110.11191) (WACV 2022)  
  [![Star](https://img.shields.io/github/stars/degardinbruno/kinetic-gan.svg?style=social&label=Star)](https://github.com/degardinbruno/kinetic-gan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.11191)

+ [Towards Using Clothes Style Transfer for Scenario-aware Person Video Generation](https://arxiv.org/pdf/2110.11894)  
  [![Star](https://img.shields.io/github/stars/xsimba123/demos-of-csf-sa.svg?style=social&label=Star)](https://github.com/xsimba123/demos-of-csf-sa)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2110.11894)

+ [Latent Image Animator: Learning to animate image via latent space navigation](https://arxiv.org/pdf/2203.09043) (ICLR 2022)  
  [![Star](https://img.shields.io/github/stars/wyhsirius/LIA.svg?style=social&label=Star)](https://github.com/wyhsirius/LIA)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09043)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://wyhsirius.github.io/LIA-project/)

+ [SLAMP: Stochastic Latent Appearance and Motion Prediction](https://arxiv.org/pdf/2108.02760) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/wyhsirius/LIA.svg?style=social&label=Star)](https://github.com/kaanakan/slamp)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.02760)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://kuis-ai.github.io/slamp/)

+ [VirtualConductor: Music-driven Conducting Video Generation System](https://arxiv.org/pdf/2108.04350) (ICME 2021)  
  [![Star](https://img.shields.io/github/stars/ChenDelong1999/VirtualConductor.svg?style=social&label=Star)](https://github.com/ChenDelong1999/VirtualConductor)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.04350)

+ [Click to Move: Controlling Video Generation with Sparse Motion](https://arxiv.org/abs/2108.08815) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/PierfrancescoArdino/C2M.svg?style=social&label=Star)](https://github.com/PierfrancescoArdino/C2M)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.08815)

+ [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/pdf/2104.10157v2.pdf)  
  [![Star](https://img.shields.io/github/stars/wilson1yan/VideoGPT.svg?style=social&label=Star)](https://github.com/wilson1yan/VideoGPT)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2104.10157v2.pdf)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://wilson1yan.github.io/videogpt/index.html)


+ [Latent Neural Differential Equations for Video Generation](https://arxiv.org/pdf/2011.03864v3.pdf)  
  [![Star](https://img.shields.io/github/stars/Zasder3/Latent-Neural-Differential-Equations-for-Video-Generation.svg?style=social&label=Star)](https://github.com/Zasder3/Latent-Neural-Differential-Equations-for-Video-Generation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2011.03864v3.pdf)

+ [Stochastic Image-to-Video Synthesis Using cINNs](https://openaccess.thecvf.com/content/CVPR2021/papers/Dorkenwald_Stochastic_Image-to-Video_Synthesis_Using_cINNs_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/CompVis/image2video-synthesis-using-cINNs.svg?style=social&label=Star)](https://github.com/CompVis/image2video-synthesis-using-cINNs)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.04551)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://compvis.github.io/image2video-synthesis-using-cINNs/)


+ [Understanding Object Dynamics for Interactive Image-to-Video Synthesis](https://openaccess.thecvf.com/content/CVPR2021/papers/Blattmann_Understanding_Object_Dynamics_for_Interactive_Image-to-Video_Synthesis_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/CompVis/interactive-image2video-synthesis.svg?style=social&label=Star)](https://github.com/CompVis/interactive-image2video-synthesis)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.11303)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://compvis.github.io/interactive-image2video-synthesis/)

+ [One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_One-Shot_Free-View_Neural_Talking-Head_Synthesis_for_Video_Conferencing_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis.svg?style=social&label=Star)](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.15126)

+ [Flow Guided Transformable Bottleneck Networks for Motion Retargeting](https://openaccess.thecvf.com/content/CVPR2021/papers/Ren_Flow_Guided_Transformable_Bottleneck_Networks_for_Motion_Retargeting_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.07771)

+ [Stable View Synthesis](https://openaccess.thecvf.com/content/CVPR2021/papers/Riegler_Stable_View_Synthesis_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/isl-org/StableViewSynthesis.svg?style=social&label=Star)](https://github.com/isl-org/StableViewSynthesis)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.07233)

+ [Scene-Aware Generative Network for Human Motion Synthesis](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Scene-Aware_Generative_Network_for_Human_Motion_Synthesis_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.14804)

+ [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Neural_Scene_Flow_Fields_for_Space-Time_View_Synthesis_of_Dynamic_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/zhengqili/Neural-Scene-Flow-Fields.svg?style=social&label=Star)](https://github.com/zhengqili/Neural-Scene-Flow-Fields)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.13084)

+ [Deep Animation Video Interpolation in the Wild](https://openaccess.thecvf.com/content/CVPR2021/papers/Siyao_Deep_Animation_Video_Interpolation_in_the_Wild_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/lisiyao21/AnimeInterp.svg?style=social&label=Star)](https://github.com/lisiyao21/AnimeInterp)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.02495)

+ [High-Fidelity Neural Human Motion Transfer from Monocular Video](https://openaccess.thecvf.com/content/CVPR2021/papers/Kappel_High-Fidelity_Neural_Human_Motion_Transfer_From_Monocular_Video_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/MoritzKappel/HF-NHMT.svg?style=social&label=Star)](https://github.com/MoritzKappel/HF-NHMT)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2012.10974)

+ [Learning Semantic-Aware Dynamics for Video Prediction](https://openaccess.thecvf.com/content/CVPR2021/papers/Bei_Learning_Semantic-Aware_Dynamics_for_Video_Prediction_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.09762)

+ [Flow-Guided One-Shot Talking Face Generation With a High-Resolution Audio-Visual Dataset](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Flow-Guided_One-Shot_Talking_Face_Generation_With_a_High-Resolution_Audio-Visual_Dataset_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/MRzzm/HDTF.svg?style=social&label=Star)](https://github.com/MRzzm/HDTF)

+ [Layout-Guided Novel View Synthesis From a Single Indoor Panorama](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Layout-Guided_Novel_View_Synthesis_From_a_Single_Indoor_Panorama_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/bluestyle97/PNVS.svg?style=social&label=Star)](https://github.com/bluestyle97/PNVS)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.17022)

+ [Space-Time Neural Irradiance Fields for Free-Viewpoint Video](https://openaccess.thecvf.com/content/CVPR2021/papers/Xian_Space-Time_Neural_Irradiance_Fields_for_Free-Viewpoint_Video_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.12950)

+ [GeoSim: Realistic Video Simulation via Geometry-Aware Composition for Self-Driving](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_GeoSim_Realistic_Video_Simulation_via_Geometry-Aware_Composition_for_Self-Driving_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2101.06543)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tmux.top/publication/geosim/)

+ [Animating Pictures With Eulerian Motion Fields](https://openaccess.thecvf.com/content/CVPR2021/papers/Holynski_Animating_Pictures_With_Eulerian_Motion_Fields_CVPR_2021_paper.pdf) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.15128)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://eulerian.cs.washington.edu/)

+ [SLAMP: Stochastic Latent Appearance and Motion Prediction](https://arxiv.org/pdf/2108.02760) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/wyhsirius/LIA.svg?style=social&label=Star)](https://github.com/kaanakan/slamp)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.02760)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://kuis-ai.github.io/slamp/)

+ [CCVS: Context-aware Controllable Video Synthesis](https://arxiv.org/pdf/2107.08037v2) (NeurIPS 2021)  
  [![Star](https://img.shields.io/github/stars/16lemoing/ccvs.svg?style=social&label=Star)](https://github.com/16lemoing/ccvs)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.08037v2)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://16lemoing.github.io/ccvs/)

+ [Diverse Video Generation using a Gaussian Process Trigger](https://arxiv.org/pdf/2107.04619) (ICLR 2021)  
  [![Star](https://img.shields.io/github/stars/shgaurav1/DVG.svg?style=social&label=Star)](https://github.com/shgaurav1/DVG)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.04619)

+ [FitVid: Overfitting in Pixel-Level Video Prediction](https://arxiv.org/pdf/2106.13195)  
  [![Star](https://img.shields.io/github/stars/google-research/fitvid.svg?style=social&label=Star)](https://github.com/google-research/fitvid)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.13195)

+ [NWT: Towards natural audio-to-video generation with representation learning](https://arxiv.org/pdf/2106.04283)  
  [![Star](https://img.shields.io/github/stars/lucidrains/NWT-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/NWT-pytorch)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.04283)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://next-week-tonight.github.io/NWT_blog/)

+ [Editable Free-viewpoint Video Using a Layered Neural Representation](https://arxiv.org/pdf/2104.14786)  
  [![Star](https://img.shields.io/github/stars/darlinghang/st-nerf.svg?style=social&label=Star)](https://github.com/darlinghang/st-nerf)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.14786)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://jiakai-zhang.github.io/st-nerf/)

+ [A Good Image Generator Is What You Need for High-Resolution Video Synthesis](https://arxiv.org/pdf/2104.15069)  
  [![Star](https://img.shields.io/github/stars/snap-research/MoCoGAN-HD.svg?style=social&label=Star)](https://github.com/snap-research/MoCoGAN-HD)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.15069)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://bluer555.github.io/MoCoGAN-HD/)

+ [GODIVA: Generating Open-DomaIn Videos from nAtural Descriptions](https://arxiv.org/pdf/2104.14806)  
  [![Star](https://img.shields.io/github/stars/mehdidc/DALLE_clip_score.svg?style=social&label=Star)](https://github.com/mehdidc/DALLE_clip_score)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.14806)

+ [Text2Video: Text-driven Talking-head Video Synthesis with Personalized Phoneme-Pose Dictionary](https://arxiv.org/pdf/2104.14631)  
  [![Star](https://img.shields.io/github/stars/sibozhang/Text2Video.svg?style=social&label=Star)](https://github.com/sibozhang/Text2Video)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2104.14631)

+ [Adaptive Appearance Rendering](https://arxiv.org/pdf/2104.11931)  
  [![Star](https://img.shields.io/github/stars/wisdomdeng/AdaptiveRendering.svg?style=social&label=Star)](https://github.com/wisdomdeng/AdaptiveRendering)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.11931)

+ [Write-a-speaker: Text-based Emotional and Rhythmic Talking-head Generation](https://arxiv.org/pdf/2104.07995)  
  [![Star](https://img.shields.io/github/stars/FuxiVirtualHuman/Write-a-Speaker.svg?style=social&label=Star)](https://github.com/FuxiVirtualHuman/Write-a-Speaker)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.07995)

+ [Predicting Video with VQVAE](https://arxiv.org/pdf/2103.01950)  
  [![Star](https://img.shields.io/github/stars/FuxiVirtualHuman/Write-a-Speaker.svg?style=social&label=Star)](https://github.com/mattiasxu/Video-VQVAE)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.01950)

+ [Playable Video Generation](https://arxiv.org/pdf/2101.12195) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/FuxiVirtualHuman/Write-a-Speaker.svg?style=social&label=Star)](https://github.com/willi-menapace/PlayableVideoGeneration)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2101.12195)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://willi-menapace.github.io/playable-video-generation-website/)

+ [Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image](https://arxiv.org/pdf/2012.09855) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/google-research/google-research.svg?style=social&label=Star)](https://github.com/google-research/google-research/tree/master/infinite_nature)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2012.09855)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://infinite-nature.github.io/)

+ [Vid-ODE: Continuous-Time Video Generation with Neural Ordinary Differential Equation](https://arxiv.org/pdf/2010.08188) (AAAI 2021)  
  [![Star](https://img.shields.io/github/stars/psh01087/Vid-ODE.svg?style=social&label=Star)](https://github.com/psh01087/Vid-ODE)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2010.08188)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://psh01087.github.io/Vid-ODE/)

+ [Compositional Video Synthesis with Action Graphs](https://arxiv.org/pdf/2006.15327) (ICML 2021)  
  [![Star](https://img.shields.io/github/stars/roeiherz/AG2Video.svg?style=social&label=Star)](https://github.com/roeiherz/AG2Video)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2006.15327)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://research.nvidia.com/labs/par/publication/sg2vid.html)

+ [Temporal Shift GAN for Large Scale Video Generation](https://arxiv.org/pdf/2004.01823) (WACV 2021)
  [![Star](https://img.shields.io/github/stars/amunozgarza/tsb-gan.svg?style=social&label=Star)](https://github.com/amunozgarza/tsb-gan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2004.01823)

+ [Learning Speech-driven 3D Conversational Gestures from Video](https://arxiv.org/pdf/2102.06837)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.06837)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vcai.mpi-inf.mpg.de/projects/3d_speech_driven_gesture/)

+ [SLPC: a VRNN-based approach for stochastic lidar prediction and completion in autonomous driving](https://arxiv.org/pdf/2102.09883)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.09883)

+ [Self-Supervision by Prediction for Object Discovery in Videos](https://arxiv.org/pdf/2103.05669)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.05669)

+ [Modulated Periodic Activations for Generalizable Local Functional Representations](https://arxiv.org/pdf/2104.03960) (ICCV 2021)
  [![Star](https://img.shields.io/github/stars/lucidrains/siren-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/siren-pytorch)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.03960)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://ishit.github.io/modsine/)

+ [Dynamic Texture Synthesis by Incorporating Long-range Spatial and Temporal Correlations](https://arxiv.org/pdf/2104.05940)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.05940)

+ [GANs N' Roses: Stable, Controllable, Diverse Image to Image Translation (works for videos too!)](https://arxiv.org/pdf/2106.06561)  
  [![Star](https://img.shields.io/github/stars/mchong6/GANsNRoses.svg?style=social&label=Star)](https://github.com/mchong6/GANsNRoses)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.06561)

+ [Alias-Free Generative Adversarial Networks](https://arxiv.org/pdf/2106.12423) (NeurIPS 2021)
  [![Star](https://img.shields.io/github/stars/NVlabs/stylegan3.svg?style=social&label=Star)](https://github.com/NVlabs/stylegan3)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.12423)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://nvlabs.github.io/stylegan3/)

+ [Modeling Clothing as a Separate Layer for an Animatable Human Avatar](https://arxiv.org/pdf/2106.14879)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.14879)

+ [CLIP-It! Language-Guided Video Summarization](https://arxiv.org/pdf/2107.00650) (NeurIPS 2021)
  [![Star](https://img.shields.io/github/stars/medhini/clip_it.svg?style=social&label=Star)](https://github.com/medhini/clip_it)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.00650)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://medhini.github.io/clip_it/)

+ [Towards an Interpretable Latent Space in Structured Models for Video Prediction](https://arxiv.org/pdf/2107.07713)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.07713)

+ [AnyoneNet: Synchronized Speech and Talking Head Generation for Arbitrary Person](https://arxiv.org/pdf/2108.04325)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.04325)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://xinshengwang.github.io/project/talking_head/)

+ [SPACE: A Simulator for Physical Interactions and Causal Learning in 3D Environments](https://arxiv.org/pdf/2108.06180)
  [![Star](https://img.shields.io/github/stars/jiafei1224/space.svg?style=social&label=Star)](https://github.com/jiafei1224/space)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.06180)

+ [PIP: Physical Interaction Prediction via Mental Simulation with Span Selection](https://arxiv.org/pdf/2109.04683)
  [![Star](https://img.shields.io/github/stars/SamsonYuBaiJian/pip.svg?style=social&label=Star)](https://github.com/SamsonYuBaiJian/pip)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.04683)

+ [Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions](https://arxiv.org/pdf/2111.10337) (CVPR 2022)  
  [![Star](https://img.shields.io/github/stars/microsoft/xpretrain.svg?style=social&label=Star)](https://github.com/microsoft/xpretrain)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.10337)

+ [Responsive Listening Head Generation: A Benchmark Dataset and Baseline](https://arxiv.org/pdf/2112.13548) (ECCV 2022)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.13548)

+ [BANMo: Building Animatable 3D Neural Models from Many Casual Videos](https://arxiv.org/pdf/2112.12761) (CVPR 2022)  
  [![Star](https://img.shields.io/github/stars/facebookresearch/banmo.svg?style=social&label=Star)](https://github.com/facebookresearch/banmo)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.12761)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://banmo-www.github.io/)

+ [Continuous-Time Video Generation via Learning Motion Dynamics with Neural ODE](https://arxiv.org/pdf/2112.10960) (BMVC 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.10960)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://psh01087.github.io/MODE-GAN/)

+ [Image Animation with Keypoint Mask](https://arxiv.org/pdf/2112.10457)  
  [![Star](https://img.shields.io/github/stars/or-toledano/animation-with-keypoint-mask.svg?style=social&label=Star)](https://github.com/or-toledano/animation-with-keypoint-mask)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.10457)

+ [SAGA: Stochastic Whole-Body Grasping with Contact](https://arxiv.org/pdf/2112.10103) (ECCV 2022)  
  [![Star](https://img.shields.io/github/stars/JiahaoPlus/SAGA.svg?style=social&label=Star)](https://github.com/JiahaoPlus/SAGA)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.10103)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://jiahaoplus.github.io/SAGA/saga.html)

+ [Adversarial Memory Networks for Action Prediction](https://arxiv.org/pdf/2112.09875)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.09875)

+ [End-to-End Rate-Distortion Optimized Learned Hierarchical Bi-Directional Video Compression](https://arxiv.org/pdf/2112.09529v1)
  [![Star](https://img.shields.io/github/stars/KUIS-AI-Tekalp-Research-Group/video-compression.svg?style=social&label=Star)](https://github.com/KUIS-AI-Tekalp-Research-Group/video-compression)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.09529)

+ [Enhanced Frame and Event-Based Simulator and Event-Based Video Interpolation Network](https://arxiv.org/pdf/2112.09379)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.09379)

+ [Discrete neural representations for explainable anomaly detection](https://arxiv.org/pdf/2112.08913) (AAAI 2022)
  [![Star](https://img.shields.io/github/stars/KT27-A/CSTP.svg?style=social&label=Star)](https://github.com/KT27-A/CSTP)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.08913)

+ [Controllable Animation of Fluid Elements in Still Images](https://arxiv.org/pdf/2112.03051) (CVPR 2022)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.03051)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://controllable-cinemagraphs.github.io/)

+ [One-shot Talking Face Generation from Single-speaker Audio-Visual Correlation Learning](https://arxiv.org/pdf/2112.02749) (AAAI 2022)
  [![Star](https://img.shields.io/github/stars/FuxiVirtualHuman/AAAI22-one-shot-talking-face.svg?style=social&label=Star)](https://github.com/FuxiVirtualHuman/AAAI22-one-shot-talking-face)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.02749)

+ [Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://arxiv.org/pdf/2112.01517) (SIGGRAPH Asia 2022)  
  [![Star](https://img.shields.io/github/stars/zju3dv/enerf.svg?style=social&label=Star)](https://github.com/zju3dv/enerf)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.01517)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://zju3dv.github.io/enerf/)

+ [Neural Point Light Fields](https://arxiv.org/abs/2112.01473) (CVPR 2022)
  [![Star](https://img.shields.io/github/stars/princeton-computational-imaging/neural-point-light-fields.svg?style=social&label=Star)](https://github.com/princeton-computational-imaging/neural-point-light-fields)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.01473)

+ [Video Frame Interpolation without Temporal Priors](https://arxiv.org/abs/2112.01161) (NeurIPS 2020)
  [![Star](https://img.shields.io/github/stars/yjzhang96/UTI-VFI.svg?style=social&label=Star)](https://github.com/yjzhang96/UTI-VFI)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.01161)

+ [ST-MFNet: A Spatio-Temporal Multi-Flow Network for Frame Interpolation](https://arxiv.org/abs/2111.15483) (CVPR 2022)
  [![Star](https://img.shields.io/github/stars/danier97/ST-MFNet.svg?style=social&label=Star)](https://github.com/danier97/ST-MFNet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.15483)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://danielism97.github.io/ST-MFNet)

+ [Video Frame Interpolation Transformer](https://arxiv.org/abs/2111.13817) (CVPR 2022)
  [![Star](https://img.shields.io/github/stars/dvlab-research/vfiformer.svg?style=social&label=Star)](https://github.com/dvlab-research/vfiformer)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.13817)

+ [Improving the Perceptual Quality of 2D Animation Interpolation](https://arxiv.org/abs/2111.12792) (ECCV 2022)
  [![Star](https://img.shields.io/github/stars/shuhongchen/eisai-anime-interpolator.svg?style=social&label=Star)](https://github.com/shuhongchen/eisai-anime-interpolator)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.12792)

+ [Layered Controllable Video Generation](https://arxiv.org/abs/2111.12747) (ECCV 2022)
  [![Star](https://img.shields.io/github/stars/Gabriel-Huang/Layered-Controllable-Video-Generation.svg?style=social&label=Star)](https://github.com/Gabriel-Huang/Layered-Controllable-Video-Generation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.12792)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://gabriel-huang.github.io/layered_controllable_video_generation/)

+ [Human Pose Manipulation and Novel View Synthesis using Differentiable Rendering](https://arxiv.org/abs/2111.12731)
  [![Star](https://img.shields.io/github/stars/guillaumerochette/humanviewsynthesis.svg?style=social&label=Star)](https://github.com/guillaumerochette/humanviewsynthesis)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.12792)

+ [Two-stage Rule-induction Visual Reasoning on RPMs with an Application to Video Prediction](https://arxiv.org/abs/2111.12301)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.12301)

+ [Video Content Swapping Using GAN](http://export.arxiv.org/abs/2111.10916)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](http://export.arxiv.org/abs/2111.10916)

+ [Temporal-MPI: Enabling Multi-Plane Images for Dynamic Scene Modelling via Temporal Basis Learning](https://arxiv.org/abs/2111.10533) (ECCV 2022)  
  [![Star](https://img.shields.io/github/stars/HKBU-VSComputing/2022_ECCV_Temporal-MPI.svg?style=social&label=Star)](https://github.com/HKBU-VSComputing/2022_ECCV_Temporal-MPI)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.10533)

+ [Xp-GAN: Unsupervised Multi-object Controllable Video Generation](https://arxiv.org/abs/2111.10233)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.10233)

+ [Action2video: Generating Videos of Human 3D Actions](https://arxiv.org/abs/2111.06925) (IJCV 2022)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.06925)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vision-and-learning-lab-ualberta.github.io/post/chuan_ijcv_2022/)

+ [Dance In the Wild: Monocular Human Animation with Neural Dynamic Appearance Synthesis](https://arxiv.org/abs/2111.05916)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.05916)

+ [LUMINOUS: Indoor Scene Generation for Embodied AI Challenges](https://arxiv.org/abs/2111.05916)  
  [![Star](https://img.shields.io/github/stars/amazon-science/indoor-scene-generation-eai.svg?style=social&label=Star)](https://github.com/amazon-science/indoor-scene-generation-eai)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.05916)

+ [FREGAN : an application of generative adversarial networks in enhancing the frame rate of videos](https://arxiv.org/abs/2111.01105)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.01105)

+ [Render In-between: Motion Guided Video Synthesis for Action Interpolation](https://arxiv.org/abs/2111.01029)  
  [![Star](https://img.shields.io/github/stars/azuxmioy/Render-In-Between.svg?style=social&label=Star)](https://github.com/azuxmioy/Render-In-Between)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.01029)

+ [Imitating Arbitrary Talking Style for Realistic Audio-DrivenTalking Face Synthesis](https://arxiv.org/abs/2111.00203) (MM 2021)  
  [![Star](https://img.shields.io/github/stars/wuhaozhe/style_avatar.svg?style=social&label=Star)](https://github.com/wuhaozhe/style_avatar)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.00203)

+ [TaylorSwiftNet: Taylor Driven Temporal Modeling for Swift Future Frame Prediction](https://arxiv.org/abs/2111.00203)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.00203)

+ [Image Comes Dancing with Collaborative Parsing-Flow Video Synthesis](https://arxiv.org/abs/2110.14147) (TIP 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.14147)

+ [H-NeRF: Neural Radiance Fields for Rendering and Temporal Reconstruction of Humans in Motion](https://arxiv.org/abs/2110.13746)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.13746)

+ [HDRVideo-GAN: Deep Generative HDR Video Reconstruction](https://arxiv.org/abs/2110.11795) (ICVGIP 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.11795)

+ [Creating and Reenacting Controllable 3D Humans with Differentiable Rendering](https://arxiv.org/abs/2110.11746) (WACV 2022)  
  [![Star](https://img.shields.io/github/stars/wuhaozhe/style_avatar.svg?style=social&label=Star)](https://github.com/verlab/CreatingAndReenacting_WACV_2022)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.11746)

+ [Wide and Narrow: Video Prediction from Context and Motion](https://arxiv.org/abs/2110.11586)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.11586)

+ [MUGL: Large Scale Multi Person Conditional Action Generation with Locomotion](https://arxiv.org/abs/2111.12792) (WACV 2022)  
  [![Star](https://img.shields.io/github/stars/skelemoa/mugl.svg?style=social&label=Star)](https://github.com/skelemoa/mugl)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.12792)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://skeleton.iiit.ac.in/mugl)

+ [LARNet: Latent Action Representation for Human Action Synthesis](https://arxiv.org/abs/2110.11236) (ICLR 2022)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.11236)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vpr-model.github.io/)

+ [Talking Head Generation with Audio and Speech Related Facial Action Units](https://arxiv.org/abs/2110.09951) (BMVC 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.09951)

+ [NeuralDiff: Segmenting 3D objects that move in egocentric videos](https://arxiv.org/abs/2110.09936) (3DV 2021)  
  [![Star](https://img.shields.io/github/stars/dichotomies/NeuralDiff.svg?style=social&label=Star)](https://github.com/dichotomies/NeuralDiff)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.09936)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.robots.ox.ac.uk/~vadim/neuraldiff/)

+ [Intelligent Video Editing: Incorporating Modern Talking Face Generation Algorithms in a Video Editor](https://arxiv.org/abs/2110.08580) (ICVGIP 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.08580)

+ [Pose-guided Generative Adversarial Net for Novel View Action Synthesis](https://arxiv.org/abs/2110.07993) (WACV 2022)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.07993)

+ [Fourier-based Video Prediction through Relational Object Motion](https://arxiv.org/abs/2110.05881)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.05881)

+ [Synthetic Data for Multi-Parameter Camera-Based Physiological Sensing](https://arxiv.org/abs/2110.04902)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.04902)

+ [Sketch Me A Video](https://arxiv.org/abs/2110.04710)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.04710)

+ [Video Autoencoder: self-supervised disentanglement of static 3D structure and motion](https://arxiv.org/abs/2110.02951) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/zlai0/VideoAutoencoder.svg?style=social&label=Star)](https://github.com/zlai0/VideoAutoencoder)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.02951)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://zlai0.github.io/VideoAutoencoder/)

+ [A Hierarchical Variational Neural Uncertainty Model for Stochastic Video Prediction](https://arxiv.org/abs/2110.03446)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.03446)

+ [Self-Supervised Decomposition, Disentanglement and Prediction of Video Sequences while Interpreting Dynamics: A Koopman Perspective](https://arxiv.org/abs/2110.00547)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.00547)

+ [A Stacking Ensemble Approach for Supervised Video Summarization](https://arxiv.org/abs/2109.12581)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.12581)

+ [Physics-based Human Motion Estimation and Synthesis from Videos](https://arxiv.org/abs/2109.09913) (ICCV 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.09913)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://nv-tlabs.github.io/physics-pose-estimation-project-page/)

+ [HYouTube: Video Harmonization Dataset](https://arxiv.org/abs/2109.08809) (Datasets)  
  [![Star](https://img.shields.io/github/stars/bcmi/video-harmonization-dataset-hyoutube.svg?style=social&label=Star)](https://github.com/bcmi/video-harmonization-dataset-hyoutube)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.08809)

+ [Diverse Generation from a Single Video Made Possible](https://arxiv.org/abs/2109.08591)  
  [![Star](https://img.shields.io/github/stars/nivha/single_video_generation.svg?style=social&label=Star)](https://github.com/nivha/single_video_generation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.08591)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://nivha.github.io/vgpnn/)

+ [Neural Human Performer: Learning Generalizable Radiance Fields for Human Performance Rendering](https://arxiv.org/abs/2109.07448)  
  [![Star](https://img.shields.io/github/stars/YoungJoongUNC/Neural_Human_Performer.svg?style=social&label=Star)](https://github.com/YoungJoongUNC/Neural_Human_Performer)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.08591)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://youngjoongunc.github.io/nhp/)

+ [Conditional MoCoGAN for Zero-Shot Video Generation](https://arxiv.org/abs/2109.05864)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.05864)

+ [Temporally Coherent Person Matting Trained on Fake-Motion Dataset](https://arxiv.org/abs/2109.04843)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.04843)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://videoprocessing.github.io/person-matting)

+ [Simple Video Generation using Neural ODEs](https://arxiv.org/abs/2109.03292)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.03292)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://voletiv.github.io/docs/presentations/20191213_Vancouver_NeurIPSW_EncODEDec.pdf)

+ [ERA: Entity Relationship Aware Video Summarization with Wasserstein GAN](https://arxiv.org/abs/2109.02625)
  [![Star](https://img.shields.io/github/stars/bcmi/video-harmonization-dataset-hyoutube.svg?style=social&label=Star)](https://github.com/jnzs1836/era-vsum)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.02625)

+ [Learning Fine-Grained Motion Embedding for Landscape Animation](https://arxiv.org/abs/2109.02216) (ACM Multimedia 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.02216)

+ [Deep Person Generation: A Survey from the Perspective of Face, Pose and Cloth Synthesis](https://arxiv.org/abs/2109.02081)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.02081)

+ [Sparse to Dense Motion Transfer for Face Image Animation](https://arxiv.org/abs/2109.00471) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/fangchangma/sparse-to-dense.svg?style=social&label=Star)](https://github.com/fangchangma/sparse-to-dense)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2109.00471)

+ [View Synthesis of Dynamic Scenes based on Deep 3D Mask Volume](https://arxiv.org/abs/2108.13408) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/ken2576/deep-3dmask.svg?style=social&label=Star)](https://github.com/ken2576/deep-3dmask)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.13408)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://cseweb.ucsd.edu//~viscomp/projects/ICCV21Deep/)

+ [Flow-Guided Video Inpainting with Scene Templates](https://arxiv.org/abs/2108.12845) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/donglao/videoinpainting.svg?style=social&label=Star)](https://github.com/donglao/videoinpainting)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.12845)

+ [Target Adaptive Context Aggregation for Video Scene Graph Generation](https://arxiv.org/abs/2108.08121) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/mcg-nju/trace.svg?style=social&label=Star)](https://github.com/mcg-nju/trace)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.08121)

+ [FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning](https://arxiv.org/abs/2108.07938) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/zhangchenxu528/FACIAL.svg?style=social&label=Star)](https://github.com/zhangchenxu528/FACIAL)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.07938)

+ [Asymmetric Bilateral Motion Estimation for Video Frame Interpolation](https://arxiv.org/abs/2108.06815) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/junheum/abme.svg?style=social&label=Star)](https://github.com/junheum/abme)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.06815)

+ [Occlusion-Aware Video Object Inpainting](https://arxiv.org/abs/2108.06765) (ICCV 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.06765)

+ [Conditional Temporal Variational AutoEncoder for Action Video Prediction](https://arxiv.org/abs/2108.05658) (ECCV 2018)  
  [![Star](https://img.shields.io/github/stars/yccyenchicheng/pytorch-VideoVAE.svg?style=social&label=Star)](https://github.com/yccyenchicheng/pytorch-VideoVAE)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.06815)

+ [UniFaceGAN: A Unified Framework for Temporally Consistent Facial Video Editing](https://arxiv.org/abs/2108.05650) (IEEE TIP 2021IEEE TIP 2021)  

  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.05650)

+ [iButter: Neural Interactive Bullet Time Generator for Human Free-viewpoint Rendering](https://arxiv.org/abs/2108.05577) (ACM MM 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.05577)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://aoliao12138.github.io/iButter/)

+ [FLAME-in-NeRF : Neural control of Radiance Fields for Free View Face Animation](https://arxiv.org/abs/2108.04913)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.04913)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://shahrukhathar.github.io/2021/08/12/FLAMEinNeRF.html)

+ [Learning to Cut by Watching Movies](https://arxiv.org/abs/2108.04294) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/PardoAlejo/LearningToCut.svg?style=social&label=Star)](https://github.com/PardoAlejo/LearningToCut)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.04294)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.alejandropardo.net/publication/learning-to-cut/)

+ [RockGPT: Reconstructing three-dimensional digital rocks from single two-dimensional slice from the perspective of video generation](https://arxiv.org/abs/2108.03132)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.03132)

+ [I2V-GAN: Unpaired Infrared-to-Visible Video Translation](https://arxiv.org/abs/2108.00913) (ACM MM 2021)  
  [![Star](https://img.shields.io/github/stars/BIT-DA/I2V-GAN.svg?style=social&label=Star)](https://github.com/BIT-DA/I2V-GAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.00913)

+ [Video Generation from Text Employing Latent Path Construction for Temporal Modeling](https://arxiv.org/abs/2107.13766)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.13766)

+ [Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion](https://arxiv.org/abs/2107.09293)  
  [![Star](https://img.shields.io/github/stars/BIT-DA/I2V-GAN.svg?style=social&label=Star)](https://github.com/wangsuzhen/Audio2Head)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.09293)

+ [Generative Video Transformer: Can Objects be the Words?](https://arxiv.org/abs/2107.09240)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.09240)

+ [StyleVideoGAN: A Temporal Generative Model using a Pretrained StyleGAN](https://arxiv.org/abs/2107.07224)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.07224)

+ [LiveView: Dynamic Target-Centered MPI for View Synthesis](https://arxiv.org/abs/2107.05113)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.05113)

+ [Speech2Video: Cross-Modal Distillation for Speech to Video Generation](https://arxiv.org/abs/2107.04806) (ACCV 2020)  
  [![Star](https://img.shields.io/github/stars/sibozhang/Speech2Video.svg?style=social&label=Star)](https://github.com/sibozhang/Speech2Video)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.04806)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/sibozhang/speech2video)

+ [Cross-View Exocentric to Egocentric Video Synthesis](https://arxiv.org/abs/2107.03120) (ACM MM 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.03120)

+ [Egocentric Videoconferencing](https://arxiv.org/abs/2107.03109)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.03109)
  [![Website](https://img.shields.io/badge/Website-9cf)](http://gvv.mpi-inf.mpg.de/projects/EgoChat/)

+ [iPOKE: Poking a Still Image for Controlled Stochastic Video Synthesis](https://arxiv.org/abs/2107.02790) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/sibozhang/Speech2Video.svg?style=social&label=Star)](https://github.com/CompVis/ipoke)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.02790)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://compvis.github.io/ipoke/)

+ [Robust Pose Transfer with Dynamic Details using Neural Video Rendering](https://arxiv.org/abs/2106.14132) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/SunYangtian/Neural-Human-Video-Rendering.svg?style=social&label=Star)](https://github.com/SunYangtian/Neural-Human-Video-Rendering)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.14132)

+ [Unsupervised Video Prediction from a Single Frame by Estimating 3D Dynamic Scene Structure](https://arxiv.org/abs/2106.09051)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.09051)

+ [Gradient Forward-Propagation for Large-Scale Temporal Video Modelling](https://arxiv.org/abs/2106.08318) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.08318)

+ [Conditional COT-GAN for Video Prediction with Kernel Smoothing](https://arxiv.org/abs/2106.05658)  
  [![Star](https://img.shields.io/github/stars/neuripss2020/kccotgan.svg?style=social&label=Star)](https://github.com/neuripss2020/kccotgan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.05658)

+ [LipSync3D: Data-Efficient Learning of Personalized 3D Talking Faces from Video using Pose and Lighting Normalization](https://arxiv.org/abs/2106.04185) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.04185)

+ [Task-Generic Hierarchical Human Motion Prior using VAEs](https://arxiv.org/abs/2106.04004)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.04004)

+ [Novel View Video Prediction Using a Dual Representation](https://arxiv.org/abs/2106.03956) (ICIP 2021)
  [![Star](https://img.shields.io/github/stars/google/stereo-magnification.svg?style=social&label=Star)](https://github.com/google/stereo-magnification)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.03956)

+ [Efficient training for future video generation based on hierarchical disentangled representation of latent variables](https://arxiv.org/abs/2106.03502)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.03502)

+ [Hierarchical Video Generation for Complex Data](https://arxiv.org/abs/2106.02719)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.02719)

+ [Temporally coherent video anonymization through GAN inpainting](https://arxiv.org/abs/2106.02328) (FG 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.02328)

+ [Anticipative Video Transformer](https://arxiv.org/abs/2106.02036) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/facebookresearch/AVT.svg?style=social&label=Star)](https://github.com/facebookresearch/AVT)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.02036)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://facebookresearch.github.io/AVT/)

+ [Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control](https://arxiv.org/abs/2106.02019) (SIGGRAPH Asia 2021)  
  [![Star](https://img.shields.io/github/stars/lingjie0206/Neural_Actor_Main_Code.svg?style=social&label=Star)](https://github.com/lingjie0206/Neural_Actor_Main_Code)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.02019)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/)

+ [Image-to-Video Generation via 3D Facial Dynamics](https://arxiv.org/abs/2105.14678)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.14678)

+ [Stylizing 3D Scene via Implicit Representation and HyperNetwork](https://arxiv.org/abs/2105.13016) (WACV 2022)  
  [![Star](https://img.shields.io/github/stars/ztex08010518/Stylizing-3D-Scene.svg?style=social&label=Star)](https://github.com/ztex08010518/Stylizing-3D-Scene)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.13016)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://ztex08010518.github.io/3dstyletransfer/)

+ [EA-Net: Edge-Aware Network for Flow-based Video Frame Interpolation](https://arxiv.org/abs/2105.07673)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.07673)

+ [Local Frequency Domain Transformer Networks for Video Prediction](https://arxiv.org/abs/2105.04637)  
  [![Star](https://img.shields.io/github/stars/AIS-Bonn/Local_Freq_Transformer_Net.svg?style=social&label=Star)](https://github.com/AIS-Bonn/Local_Freq_Transformer_Net)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.04637)

+ [Stochastic Image-to-Video Synthesis using cINNs](https://arxiv.org/abs/2105.04551) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/CompVis/image2video-synthesis-using-cINNs.svg?style=social&label=Star)](https://github.com/CompVis/image2video-synthesis-using-cINNs)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.04551)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://compvis.github.io/image2video-synthesis-using-cINNs/)

+ [Reconstructive Sequence-Graph Network for Video Summarization](https://arxiv.org/abs/2105.04066) (IEEE TPAMI 2021)   
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.04066)

+ [Object-centric Video Prediction without Annotation](https://arxiv.org/abs/2105.02799)  
  [![Star](https://img.shields.io/github/stars/kschmeckpeper/opa.svg?style=social&label=Star)](https://github.com/kschmeckpeper/opa)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.02799)

+ [Pose-Guided Sign Language Video GAN with Dynamic Lambda](https://arxiv.org/abs/2105.02742)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.02742)

+ [Moving SLAM: Fully Unsupervised Deep Learning in Non-Rigid Scenes](https://arxiv.org/abs/2105.02195)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.02195)

+ [VCGAN: Video Colorization with Hybrid Generative Adversarial Network](https://arxiv.org/abs/2104.12357) (IEEE (TMM)2021)
  [![Star](https://img.shields.io/github/stars/zhaoyuzhi/VCGAN.svg?style=social&label=Star)](https://github.com/zhaoyuzhi/VCGAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.12357)

+ [3D-TalkEmo: Learning to Synthesize 3D Emotional Talking Head](https://arxiv.org/abs/2104.12051)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.12051)

+ [Hierarchical Motion Understanding via Motion Programs](https://arxiv.org/abs/2104.11216) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/Sumith1896/motion2prog_release.svg?style=social&label=Star)](https://github.com/Sumith1896/motion2prog_release)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.11216)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sumith1896.github.io/motion2prog/)

+ [Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation](https://arxiv.org/abs/2104.11116) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/Hangz-nju-cuhk/Talking-Face_PC-AVS.svg?style=social&label=Star)](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.11116)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://hangz-nju-cuhk.github.io/projects/PC-AVS)

+ [Learning Semantic-Aware Dynamics for Video Prediction](https://arxiv.org/abs/2104.09762) (CVPR 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.09762)

+ [MeshTalk: 3D Face Animation from Speech using Cross-Modality Disentanglement](https://arxiv.org/abs/2104.09762) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/facebookresearch/meshtalk.svg?style=social&label=Star)](https://github.com/facebookresearch/meshtalk)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.09762)

+ [Zooming SlowMo: An Efficient One-Stage Framework for Space-Time Video Super-Resolution](https://arxiv.org/abs/2104.07473) (CVPR 2020)  
  [![Star](https://img.shields.io/github/stars/Mukosame/Zooming-Slow-Mo-CVPR-2020.svg?style=social&label=Star)](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.07473)

+ [Audio-Driven Emotional Video Portraits](https://arxiv.org/abs/2104.07452) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/jixinya/EVP.svg?style=social&label=Star)](https://github.com/jixinya/EVP)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.07452)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://jixinya.github.io/projects/evp/)

+ [Revisiting Hierarchical Approach for Persistent Long-Term Video Prediction](https://arxiv.org/abs/2104.06697) (ICLR 2021)  
  [![Star](https://img.shields.io/github/stars/1Konny/HVP.svg?style=social&label=Star)](https://github.com/1Konny/HVP)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.06697)

+ [Strumming to the Beat: Audio-Conditioned Contrastive Video Textures](https://arxiv.org/abs/2104.02687) (WACV 2022)  
  [![Star](https://img.shields.io/github/stars/medhini/audio-video-textures.svg?style=social&label=Star)](https://github.com/medhini/audio-video-textures)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.02687)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://medhini.github.io/audio_video_textures/)

+ [PDWN: Pyramid Deformable Warping Network for Video Interpolation](https://arxiv.org/abs/2104.01517)
  [![Star](https://img.shields.io/github/stars/zhiqiiiiiii/PDWN_for_Video_Interp.svg?style=social&label=Star)](https://github.com/zhiqiiiiiii/PDWN_for_Video_Interp)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.01517)

+ [M3L: Language-based Video Editing via Multi-Modal Multi-Level Transformers](https://arxiv.org/abs/2104.01122) (CVPR 2022)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.01122)

+ [Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning](https://arxiv.org/abs/2104.00924) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/zhiqiiiiiii/PDWN_for_Video_Interp.svg?style=social&label=Star)](https://github.com/sangmin-git/LMC-Memory)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.00924)

+ [Collaborative Learning to Generate Audio-Video Jointly](https://arxiv.org/abs/2104.02656) (ICASSP 2021)  
  [![Star](https://img.shields.io/github/stars/DelTA-Lab-IITK/AVG.svg?style=social&label=Star)](https://github.com/DelTA-Lab-IITK/AVG)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.02656)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://delta-lab-iitk.github.io/AVG/)

+ [Long-Term Temporally Consistent Unpaired Video Translation from Simulated Surgical 3D Data](https://arxiv.org/abs/2103.17204) (IJCV 2021)  
  [![Star](https://img.shields.io/github/stars/verlab/ShapeAwareHumanRetargeting_IJCV_2021.svg?style=social&label=Star)](https://github.com/verlab/ShapeAwareHumanRetargeting_IJCV_2021)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.17204)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://verlab.github.io/ShapeAwareHumanRetargeting_IJCV_2021/)

+ [A Shape-Aware Retargeting Approach to Transfer Human Motion and Appearance in Monocular Videos](https://arxiv.org/abs/2103.15596) (IJCV 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.15596)

+ [AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis](https://arxiv.org/abs/2103.11078) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/YudongGuo/AD-NeRF.svg?style=social&label=Star)](https://github.com/YudongGuo/AD-NeRF)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.11078)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yudongguo.github.io/ADNeRF/)

+ [Future Frame Prediction for Robot-assisted Surgery](https://arxiv.org/abs/2103.10308) (IPMI 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.10308)

+ [Learning to compose 6-DoF omnidirectional videos using multi-sphere images](https://arxiv.org/abs/2103.05842)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.05842)

+ [Behavior-Driven Synthesis of Human Dynamics](https://arxiv.org/abs/2103.04677) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/CompVis/behavior-driven-video-synthesis.svg?style=social&label=Star)](https://github.com/CompVis/behavior-driven-video-synthesis)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.04677)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://compvis.github.io/behavior-driven-video-synthesis/)

+ [Greedy Hierarchical Variational Autoencoders for Large-Scale Video Prediction](https://arxiv.org/abs/2103.04174)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.04174)

+ [Motion-blurred Video Interpolation and Extrapolation](https://arxiv.org/abs/2103.02984) (AAAI 2021)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.02984)

+ [Neural 3D Video Synthesis from Multi-view Video](https://arxiv.org/abs/2103.02597) (CVPR 2022)  
  [![Star](https://img.shields.io/github/stars/facebookresearch/neural_3d_video.svg?style=social&label=Star)](https://github.com/facebookresearch/neural_3d_video)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.02597)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://neural-3d-video.github.io/)

+ [MotionRNN: A Flexible Model for Video Prediction with Spacetime-Varying Motions](https://arxiv.org/abs/2103.02243) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/facebookresearch/neural_3d_video  .svg?style=social&label=Star)](https://github.com/thuml/MotionRNN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.02243)

+ [Dual-MTGAN: Stochastic and Deterministic Motion Transfer for Image-to-Video Synthesis](https://arxiv.org/abs/2102.13329) (ICPR 2020)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.13329)

+ [One Shot Audio to Animated Video Generation](https://arxiv.org/abs/2102.09737)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.09737)

+ [Clockwork Variational Autoencoders](https://arxiv.org/abs/2102.09532)  
  [![Star](https://img.shields.io/github/stars/vaibhavsaxena11/cwvae.svg?style=social&label=Star)](https://github.com/vaibhavsaxena11/cwvae)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.09532)

+ [Frame Difference-Based Temporal Loss for Video Stylization](https://arxiv.org/abs/2102.05822)  
  [![Star](https://img.shields.io/github/stars/AtlantixJJ/frame-difference-loss.svg?style=social&label=Star)](https://github.com/AtlantixJJ/frame-difference-loss)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.05822)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://atlantixjj.github.io/FDB/)
  
+ [TräumerAI: Dreaming Music with StyleGAN](https://arxiv.org/abs/2102.04680) (NeurIPS Workshop 2020)  
  [![Star](https://img.shields.io/github/stars/jdasam/traeumerAI.svg?style=social&label=Star)](https://github.com/jdasam/traeumerAI)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.04680)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://jdasam.github.io/traeumerAI_demo/)

+ [Self-Supervised Equivariant Scene Synthesis from Video](https://arxiv.org/abs/2102.00863)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.00863)

+ [VAE^2: Preventing Posterior Collapse of Variational Video Predictions in the Wild](https://arxiv.org/abs/2101.12050)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2101.12050)

+ [AI Choreographer: Music Conditioned 3D Dance Generation with AIST++](https://arxiv.org/abs/2101.08779)  
  [![Star](https://img.shields.io/github/stars/google-research/mint.svg?style=social&label=Star)](https://github.com/google-research/mint)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.04680)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://google.github.io/aichoreographer/)

+ [Disentangled Recurrent Wasserstein Autoencoder](https://arxiv.org/abs/2101.07496)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2101.07496)

+ [ArrowGAN : Learning to Generate Videos by Learning Arrow of Time](https://arxiv.org/abs/2101.03710)
  [![Star](https://img.shields.io/github/stars/jdasam/traeumerAI.svg?style=social&label=Star)](https://github.com/Kibeom-Hong/ArrowGAN-pytorch)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2101.03710)

+ [InMoDeGAN: Interpretable Motion Decomposition Generative Adversarial Network for Video Generation](https://arxiv.org/abs/2101.03049)  
  [![Star](https://img.shields.io/github/stars/c1a1o1/InMoDeGAN-project.svg?style=social&label=Star)](https://github.com/c1a1o1/InMoDeGAN-project)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2101.03049)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://wyhsirius.github.io/InMoDeGAN/)

+ [Personal Privacy Protection via Irrelevant Faces Tracking and Pixelation in Video Live Streaming](https://arxiv.org/abs/1903.10836)  
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1903.10836)

+ [Dynamic View Synthesis from Dynamic Monocular Video](https://arxiv.org/abs/2105.06468) (ICCV 2021)  
  [![Star](https://img.shields.io/github/stars/gaochen315/DynamicNeRF.svg?style=social&label=Star)](https://github.com/gaochen315/DynamicNeRF)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.06468)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://free-view-video.github.io/)

+ [Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Dynamic Scene From Monocular Video](https://arxiv.org/abs/2012.12247) (CVPR 2021)  
  [![Star](https://img.shields.io/github/stars/facebookresearch/nonrigid_nerf.svg?style=social&label=Star)](https://github.com/facebookresearch/nonrigid_nerf)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2012.12247)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vcai.mpi-inf.mpg.de/projects/nonrigid_nerf/)



## 2020

+ [Stochastic Talking Face Generation Using Latent Distribution Matching](https://arxiv.org/pdf/2011.10727)  
  [![Star](https://img.shields.io/github/stars/ry85/Stochastic-Talking-Face-Generation-Using-Latent-Distribution-Matching.svg?style=social&label=Star)](https://github.com/ry85/Stochastic-Talking-Face-Generation-Using-Latent-Distribution-Matching)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.10727)

+ [Latent Neural Differential Equations for Video Generation](https://arxiv.org/pdf/2011.03864)  
  [![Star](https://img.shields.io/github/stars/Zasder3/Latent-Neural-Differential-Equations-for-Video-Generation.svg?style=social&label=Star)](https://github.com/Zasder3/Latent-Neural-Differential-Equations-for-Video-Generation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.03864)

+ [LIFI: Towards Linguistically Informed Frame Interpolation](https://arxiv.org/pdf/2010.16078)  
  [![Star](https://img.shields.io/github/stars/midas-research/linguistically-informed-frame-interpolation.svg?style=social&label=Star)](https://github.com/midas-research/linguistically-informed-frame-interpolation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2010.16078)

+ [Unsupervised object-centric video generation and decomposition in 3D](https://arxiv.org/pdf/2007.06705) (NeurIPS 2020)  
  [![Star](https://img.shields.io/github/stars/pmh47/o3v.svg?style=social&label=Star)](https://github.com/pmh47/o3v)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2007.06705)

+ [Novel-View Human Action Synthesis](https://arxiv.org/pdf/2007.02808) (ACCV 2020)  
  [![Star](https://img.shields.io/github/stars/mlakhal/gtnet.svg?style=social&label=Star)](https://github.com/mlakhal/gtnet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2007.02808)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://mlakhal.github.io/novel-view_action_synthesis.html)

+ [Structure-Aware Human-Action Generation](https://arxiv.org/pdf/2007.01971) (ECCV 2020)  
  [![Star](https://img.shields.io/github/stars/PingYu-iris/SA-GCN.svg?style=social&label=Star)](https://github.com/PingYu-iris/SA-GCN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2007.01971)

+ [Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample](https://arxiv.org/pdf/2006.12226) (NeurIPS 2020)  
  [![Star](https://img.shields.io/github/stars/shirgur/hp-vae-gan.svg?style=social&label=Star)](https://github.com/shirgur/hp-vae-gan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2006.12226)

+ [Latent Video Transformer](https://arxiv.org/pdf/2006.10704)  
  [![Star](https://img.shields.io/github/stars/rakhimovv/lvt.svg?style=social&label=Star)](https://github.com/rakhimovv/lvt)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2006.10704)

+ [Audio-driven Talking Face Video Generation with Learning-based Personalized Head Pose](https://arxiv.org/pdf/2002.10137)  
  [![Star](https://img.shields.io/github/stars/yiranran/Audio-driven-TalkingFace-HeadPose.svg?style=social&label=Star)](https://github.com/yiranran/Audio-driven-TalkingFace-HeadPose)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2002.10137)

+ [Exploring Spatial-Temporal Multi-Frequency Analysis for High-Fidelity and Temporal-Consistency Video Prediction](https://arxiv.org/pdf/2002.09905) (CVPR 2020)  
  [![Star](https://img.shields.io/github/stars/Bei-Jin/STMFANet.svg?style=social&label=Star)](https://github.com/Bei-Jin/STMFANet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2002.09905)

+ [Stochastic Latent Residual Video Prediction](https://arxiv.org/pdf/2002.09219) (ICML 2020)  
  [![Star](https://img.shields.io/github/stars/edouardelasalles/srvp.svg?style=social&label=Star)](https://github.com/edouardelasalles/srvp)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2002.09219)

+ [G3AN: Disentangling Appearance and Motion for Video Generation](https://arxiv.org/pdf/1912.05523) (CVPR 2020)  
  [![Star](https://img.shields.io/github/stars/wyhsirius/g3an-project.svg?style=social&label=Star)](https://github.com/wyhsirius/g3an-project)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1912.05523)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://wyhsirius.github.io/G3AN/)

+ [Scaling Autoregressive Video Models](https://arxiv.org/pdf/1906.02634) (ICLR 2020)  
  [![Star](https://img.shields.io/github/stars/rakhimovv/lvt.svg?style=social&label=Star)](https://github.com/rakhimovv/lvt)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1906.02634)

+ [VideoFlow: A Conditional Flow-Based Model for Stochastic Video Generation](https://arxiv.org/pdf/1903.04480) (ICLR 2020)
  [![Star](https://img.shields.io/github/stars/tensorflow/tensor2tensor.svg?style=social&label=Star)](https://github.com/tensorflow/tensor2tensor)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1903.04480)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://lucassheng.github.io/publication/pan-video-2019/)

## 2019

+ [Music-oriented Dance Video Synthesis with Pose Perceptual Loss](https://arxiv.org/pdf/2002.09219)  
  [![Star](https://img.shields.io/github/stars/xrenaa/Music-Dance-Video-Synthesis.svg?style=social&label=Star)](https://github.com/xrenaa/Music-Dance-Video-Synthesis)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2002.09219)

+ [DwNet: Dense warp-based network for pose-guided human video generation](https://arxiv.org/pdf/1910.09139)  
  [![Star](https://img.shields.io/github/stars/ubc-vision/DwNet.svg?style=social&label=Star)](https://github.com/ubc-vision/DwNet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1910.09139)

+ [Order Matters: Shuffling Sequence Generation for Video Prediction](https://arxiv.org/pdf/1907.08845)  
  [![Star](https://img.shields.io/github/stars/andrewjywang/SEENet.svg?style=social&label=Star)](https://github.com/andrewjywang/SEENet)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1907.08845)

+ [Adversarial Video Generation on Complex Datasets](https://arxiv.org/pdf/1907.06571)  
  [![Star](https://img.shields.io/github/stars/Harrypotterrrr/DVD-GAN.svg?style=social&label=Star)](https://github.com/Harrypotterrrr/DVD-GAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1907.06571)

+ [From Here to There: Video Inbetweening Using Direct 3D Convolutions](https://arxiv.org/pdf/1905.10240)  
  [![Star](https://img.shields.io/github/stars/xih108/Video_Completion.svg?style=social&label=Star)](https://github.com/xih108/Video_Completion)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1905.10240)

+ [Improved Conditional VRNNs for Video Prediction](https://arxiv.org/pdf/1904.12165) (ICCV 2019)  
  [![Star](https://img.shields.io/github/stars/facebookresearch/improved_vrnn.svg?style=social&label=Star)](https://github.com/facebookresearch/improved_vrnn)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1904.12165)

+ [Sliced Wasserstein Generative Models](https://arxiv.org/pdf/1706.02631) (CVPR 2019)  
  [![Star](https://img.shields.io/github/stars/musikisomorphie/swd.svg?style=social&label=Star)](https://github.com/musikisomorphie/swd)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1706.02631)

+ [Point-to-Point Video Generation](https://arxiv.org/pdf/1904.02912) (ICCV 2019)  
  [![Star](https://img.shields.io/github/stars/yccyenchicheng/p2pvg.svg?style=social&label=Star)](https://github.com/yccyenchicheng/p2pvg)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1904.02912)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://zswang666.github.io/P2PVG-Project-Page/)

+ [High Frame Rate Video Reconstruction based on an Event Camera](https://arxiv.org/pdf/1903.06531)  
  [![Star](https://img.shields.io/github/stars/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera.svg?style=social&label=Star)](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1903.06531)

+ [Video Generation from Single Semantic Label Map](https://arxiv.org/pdf/1903.04480) (CVPR 2019)  
  [![Star](https://img.shields.io/github/stars/junting/seg2vid.svg?style=social&label=Star)](https://github.com/junting/seg2vid)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1903.04480)

+ [Learning to navigate image manifolds induced by generative adversarial networks for unsupervised video generation](https://arxiv.org/pdf/1901.11384)  
  [![Star](https://img.shields.io/github/stars/junting/seg2vid.svg?style=social&label=Star)](https://github.com/belaalb/frameGAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1901.11384)

+ [Animating Arbitrary Objects via Deep Motion Transfer](https://arxiv.org/pdf/1812.08861) (CVPR 2019)  
  [![Star](https://img.shields.io/github/stars/AliaksandrSiarohin/monkey-net.svg?style=social&label=Star)](https://github.com/AliaksandrSiarohin/monkey-net)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1812.08861)

+ [StoryGAN: A Sequential Conditional GAN for Story Visualization](https://arxiv.org/pdf/1812.02784) (CVPR 2019)  
  [![Star](https://img.shields.io/github/stars/yitong91/StoryGAN.svg?style=social&label=Star)](https://github.com/yitong91/StoryGAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1812.02784)

+ [Stochastic Adversarial Video Prediction](https://arxiv.org/pdf/1804.01523) (ICLR 2019)  
  [![Star](https://img.shields.io/github/stars/alexlee-gk/video_prediction.svg?style=social&label=Star)](https://github.com/alexlee-gk/video_prediction)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1804.01523)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://video-prediction.github.io/video_prediction/)

## 2018

+ [TwoStreamVAN: Improving Motion Modeling in Video Generation](https://arxiv.org/pdf/1812.01037)  
  [![Star](https://img.shields.io/github/stars/sunxm2357/TwoStreamVAN.svg?style=social&label=Star)](https://github.com/sunxm2357/TwoStreamVAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1812.01037)

+ [Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation](https://arxiv.org/pdf/1811.09393)  
  [![Star](https://img.shields.io/github/stars/thunil/TecoGAN.svg?style=social&label=Star)](https://github.com/thunil/TecoGAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1811.09393)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://ge.in.tum.de/wp-content/uploads/2020/05/ClickMe.html)

+ [Towards High Resolution Video Generation with Progressive Growing of Sliced Wasserstein GANs](https://arxiv.org/pdf/1810.02419)  
  [![Star](https://img.shields.io/github/stars/musikisomorphie/swd.svg?style=social&label=Star)](https://github.com/musikisomorphie/swd)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1810.02419)

+ [Everybody Dance Now](https://arxiv.org/pdf/1808.07371) (ICCV 2019)  
  [![Star](https://img.shields.io/github/stars/carolineec/EverybodyDanceNow.svg?style=social&label=Star)](https://github.com/carolineec/EverybodyDanceNow)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1808.07371)

+ [Learning to Forecast and Refine Residual Motion for Image-to-Video Generation](https://arxiv.org/pdf/1807.09951) (ECCV 2018)  
  [![Star](https://img.shields.io/github/stars/garyzhao/FRGAN.svg?style=social&label=Star)](https://github.com/garyzhao/FRGAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1807.09951)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://garyzhao.github.io/archives/eccv18_frgan_poster.pdf)

+ [Talking Face Generation by Conditional Recurrent Adversarial Network](https://arxiv.org/pdf/1804.04786)  
  [![Star](https://img.shields.io/github/stars/susanqq/Talking_Face_Generation.svg?style=social&label=Star)](https://github.com/susanqq/Talking_Face_Generation)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1804.04786)

+ [Probabilistic Video Generation using Holistic Attribute Control](https://arxiv.org/pdf/1803.08085) (ECCV 2018)  
  [![Star](https://img.shields.io/github/stars/yccyenchicheng/pytorch-VideoVAE.svg?style=social&label=Star)](https://github.com/yccyenchicheng/pytorch-VideoVAE)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1803.08085)

+ [Stochastic Video Generation with a Learned Prior](https://arxiv.org/pdf/1802.07687) (ICML 2018)  
  [![Star](https://img.shields.io/github/stars/edenton/svg.svg?style=social&label=Star)](https://github.com/edenton/svg)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1802.07687)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://holmdk.github.io/2020/01/22/stochastic_vid.html)

+ [Stochastic Video Generation with a Learned Prior](https://arxiv.org/pdf/1802.07687) (ICML 2018)  
  [![Star](https://img.shields.io/github/stars/edenton/svg.svg?style=social&label=Star)](https://github.com/edenton/svg)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1802.07687)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://holmdk.github.io/2020/01/22/stochastic_vid.html)

+ [Stochastic Variational Video Prediction](https://arxiv.org/pdf/1710.11252) (ICLR 2018)  
  [![Star](https://img.shields.io/github/stars/RoboTurk-Platform/roboturk_real_dataset.svg?style=social&label=Star)](https://github.com/RoboTurk-Platform/roboturk_real_dataset)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1710.11252)

+ [Hierarchical Video Generation from Orthogonal Information: Optical Flow and Texture](https://arxiv.org/pdf/1711.09618) (AAAI 2018)  
  [![Star](https://img.shields.io/github/stars/mil-tokyo/FTGAN.svg?style=social&label=Star)](https://github.com/mil-tokyo/FTGAN)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1711.09618)

+ [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/pdf/1707.04993) (CVPR 2018)  
  [![Star](https://img.shields.io/github/stars/sergeytulyakov/mocogan.svg?style=social&label=Star)](https://github.com/sergeytulyakov/mocogan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1707.04993)

## 2017

+ [Improving Video Generation for Multi-functional Applications](https://arxiv.org/pdf/1711.11453)  
  [![Star](https://img.shields.io/github/stars/bernhard2202/improved-video-gan.svg?style=social&label=Star)](https://github.com/bernhard2202/improved-video-gan)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1711.11453)

+ [Attentive Semantic Video Generation using Captions](https://arxiv.org/pdf/1708.05980) (ICCV 2017)  
  [![Star](https://img.shields.io/github/stars/Singularity42/cap2vid.svg?style=social&label=Star)](https://github.com/Singularity42/cap2vid)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1708.05980)

+ [Temporal Generative Adversarial Nets with Singular Value Clipping](https://arxiv.org/pdf/1611.06624) (ICCV 2017)  
  [![Star](https://img.shields.io/github/stars/universome/stylegan-v.svg?style=social&label=Star)](https://github.com/universome/stylegan-v)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1611.06624)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://pfnet-research.github.io/tgan/)


## 2016

+ [Sync-DRAW: Automatic Video Generation using Deep Recurrent Attentive Architectures](https://arxiv.org/pdf/1611.10314)  
  [![Star](https://img.shields.io/github/stars/Singularity42/Sync-DRAW.svg?style=social&label=Star)](https://github.com/Singularity42/Sync-DRAW)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1611.10314)

+ [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/pdf/1605.07157)  
  [![Star](https://img.shields.io/github/stars/Xiaohui9607/physical_interaction_video_prediction_pytorch.svg?style=social&label=Star)](https://github.com/Xiaohui9607/physical_interaction_video_prediction_pytorch)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1605.07157)

<!--
  // Entry template

+ [TITLE](LINK) (CONFERENCE)  
  [![Star](https://img.shields.io/github/stars/XXX/YYY.svg?style=social&label=Star)](GITHUB)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](ARXIV)
  [![Website](https://img.shields.io/badge/Website-9cf)](WEBSITE)

-->
