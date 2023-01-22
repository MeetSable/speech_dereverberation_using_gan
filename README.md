# Speech Dereverberation using Generative Adversarial Training

- Implemented in python with help of pytorch.
- [Technical Report](Technical_Report.pdf) of this project
- [Reference Paper](https://drive.google.com/file/d/1igCjRad2nnB9fklrpvN7WoNLtLfjBTve/view?usp=sharing)

## Description
- [helper_funcs.py](helper_funcs.py) contains functions like wave_to_stft, stft_to_wave, convolution, etc
- [dereverberator_gan.py](dereverberator_gan.py) Model class definitions for Generator and Discriminator and training function for the GAN
- [Kaggle Page](https://www.kaggle.com/code/meetsable/dereverberation-with-gat/notebook)

## Results
- Clean Speech

https://user-images.githubusercontent.com/53657302/213902735-5db42fa9-3111-42f9-8e89-558df145c306.mp4

-Reverberated Speech

https://user-images.githubusercontent.com/53657302/213902783-34e96c8f-4fc9-42da-a8ff-1040ee74ab7b.mp4

- Filtered with GAN (25k iters)

https://user-images.githubusercontent.com/53657302/213902789-69537b50-e7be-4190-9857-1f897f7efc0b.mp4
