# Music Mixing Style Transfer

This repository includes source code and pre-trained models of the work *Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects* by [Junghyun Koo](https://linkedin.com/in/junghyun-koo-525a31251), [Marco A. Martínez-Ramírez](https://m-marco.com/about/), [Wei-Hsiang Liao](https://jp.linkedin.com/in/wei-hsiang-liao-66283154), [Stefan Uhlich](https://scholar.google.de/citations?user=hja8ejYAAAAJ&hl=de), [Kyogu Lee](https://linkedin.com/in/kyogu-lee-7a93b611), and [Yuki Mitsufuji](https://www.yukimitsufuji.com/).


[![arXiv](https://img.shields.io/badge/arXiv-2211.02247-b31b1b.svg)](https://arxiv.org/abs/2211.02247)
[![Web](https://img.shields.io/badge/Web-Demo_Page-green.svg)](https://jhtonyKoo.github.io/MixingStyleTransfer/)
[![Supplementary](https://img.shields.io/badge/Supplementary-Materials-white.svg)](https://tinyurl.com/4math4pm)



## Pre-trained Models
| Model | Configuration | Training Dataset |
|-------------|-------------|-------------|
[FXencoder (Φ<sub>p.s.</sub>)](https://drive.google.com/file/d/1BFABsJRUVgJS5UE5iuM03dbfBjmI9LT5/view?usp=sharing) | Used *FX normalization* and *probability scheduling* techniques for training | Trained with [MUSDB18](https://sigsep.github.io/datasets/musdb.html) Dataset
[MixFXcloner](https://drive.google.com/file/d/1Qu8rD7HpTNA1gJUVp2IuaeU_Nue8-VA3/view?usp=sharing) | Mixing style converter trained with Φ<sub>p.s.</sub> | Trained with [MUSDB18](https://sigsep.github.io/datasets/musdb.html) Dataset



# Inference

## Mixing Style Transfer

To run the inference code for <i>mixing style transfer</i>, 
1. Download pre-trained models above and place them under the folder named 'weights' (default)
2. Prepare input and reference tracks under the folder named 'samples/style_transfer' (default)
Target files should be organized as follow:
```
    "path_to_data_directory"/"song_name_#1"/"input_file_name".wav
    "path_to_data_directory"/"song_name_#1"/"reference_file_name".wav
    ...
    "path_to_data_directory"/"song_name_#n"/"input_file_name".wav
    "path_to_data_directory"/"song_name_#n"/"reference_file_name".wav
```
3. Run 'inference/style_transfer.py'
```
python inference/style_transfer.py \
    --ckpt_path_enc "path_to_checkpoint_of_FXencoder" \
    --ckpt_path_conv "path_to_checkpoint_of_MixFXcloner" \
    --target_dir "path_to_directory_containing_inference_samples"
```
4. Outputs will be stored under the same folder to inference data directory (default)

*Note: The system accepts WAV files of stereo-channeled, 44.1kHZ, and 16-bit rate. We recommend to use audio samples that are not too loud: it's better for the system to transfer these samples by reducing the loudness of mixture-wise inputs (maintaining the overall balance of each instrument).*



## Interpolation With 2 Different Reference Tracks

Inference code for <interpolating> two reference tracks is almost the same as <i>mixing style transfer</i>.
1. Download pre-trained models above and place them under the folder named 'weights' (default)
2. Prepare input and 2 reference tracks under the folder named 'samples/style_transfer' (default)
Target files should be organized as follow:
```
    "path_to_data_directory"/"song_name_#1"/"input_track_name".wav
    "path_to_data_directory"/"song_name_#1"/"reference_file_name".wav
    "path_to_data_directory"/"song_name_#1"/"reference_file_name_2interpolate".wav
    ...
    "path_to_data_directory"/"song_name_#n"/"input_track_name".wav
    "path_to_data_directory"/"song_name_#n"/"reference_file_name".wav
    "path_to_data_directory"/"song_name_#n"/"reference_file_name_2interpolate".wav
```
3. Run 'inference/style_transfer.py'
```
python inference/style_transfer.py \
    --ckpt_path_enc "path_to_checkpoint_of_FXencoder" \
    --ckpt_path_conv "path_to_checkpoint_of_MixFXcloner" \
    --target_dir "path_to_directory_containing_inference_samples" \
    --interpolation True \
    --interpolate_segments "number of segments to perform interpolation"
```
4. Outputs will be stored under the same folder to inference data directory (default)

*Note: This example of interpolating 2 different reference tracks is not mentioned in the paper, but this example implies a potential for controllable style transfer using latent space.*



## Feature Extraction Using *FXencoder*

This inference code will extracts audio effects-related embeddings using our proposed <i>FXencoder</i>. This code will process all the .wav files under the target directory.

1. Download <i>FXencoder</i>'s pre-trained model above and place it under the folder named 'weights' (default)=
2. Run 'inference/style_transfer.py'
```
python inference/feature_extraction.py \
    --ckpt_path_enc "path_to_checkpoint_of_FXencoder" \
    --target_dir "path_to_directory_containing_inference_samples"
```
3. Outputs will be stored under the same folder to inference data directory (default)




# Implementation

All the details of our system implementation are under the folder "mixing_style_transfer".

<li><i>FXmanipulator</i></li>
&emsp;&emsp;-> mixing_style_transfer/mixing_manipulator/
<li>network architectures</li>
&emsp;&emsp;-> mixing_style_transfer/networks/
<li>configuration of each sub-networks</li>
&emsp;&emsp;-> mixing_style_transfer/networks/configs.yaml
<li>data loader</li>
&emsp;&emsp;-> mixing_style_transfer/data_loader/




