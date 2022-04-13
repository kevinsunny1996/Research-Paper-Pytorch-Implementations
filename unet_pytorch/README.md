# Implementation of U-Net Image Segmentation deep learning architecture using Pytorch 🥳

## Credits for paper and code implementation

 - 🤓 [U-Net: Convolutional Networks for Biomedical Image Segmentation - Olaf Ronneberger, Philipp Fischer, and Thomas Brox](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical) 

 - 👨‍💻 [Pytorch implementation by Aladdin Persson](https://www.youtube.com/watch?v=IHq1t7NxS8k)


 ## Key takeaways from the 2015 paper 🛒
- U-Net is a powerful architecture for **image segmentation**.
- The paper is a good introduction to the architecture and the basic concepts which was used for biomedical image segmentation.
- The pytorch implementation of this goes a bit further by using the same concepts for segmenting car images based on their masks.
- In cases where the problem involves identifying multiple objects ,as this architecture proves to be better at localizing them.
- This **localization technique** helps in better **capturing context** aiding in better segmentation of the images that will be provided to this model when live.
 ## Initial challenges 😣 and how were they solved 😌 !!!
 - [x] (Solved) Running train.py resulted in FileNotFound Exception at Data Loader end.
    - Fixed using the correct dataset as the code was supposed to work with _mask.gif files for masks.
    - Refer link while downloading dataset:
        - [Download only train.zip and train_masks.zip](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)
- [ ] (Ongoing) Segmentation fault received when running train.py
    - In such scenario, when the debugging needs to be done , use gdb (apt install gdb) "depending on linux distro" to get stacktrace.
    - Syntax - `gdb train.py`.
    - In this case , the error was occuring at libtorch_cpu.so which is being investigated further.
 ## Learnings from implementing the paper 🕵🏽
- Learnt how to better use OOP while implementing an ML algo which is very rare to find.
- Learnt how to better modularize your code.
- Learnt how to better divide the layers and convert them into solutions for the U-Net architecture.

 ## Parting words 👋