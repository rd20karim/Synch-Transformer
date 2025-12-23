## Description

Official implementation of Synch-Transformer for synchronous motion captioning:

<div align="center">

[<span style="font-size: 25px;"> <span style="color:darkviolet; font-weight:darkbold; font-size: 30px">Transformer with Controlled Attention for Synchronous Captioning </span>](https://hal.science/hal-04697946)

[![arxiv](https://img.shields.io/badge/arXiv-Synch_Transformer-cyan?logo=arxiv)](http://arxiv.org/abs/2409.09177)
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

This work introduces a Transformer-based design to address the task of motion-to-text synchronization, as introduced in this previous project [m2t-segmentation](https://github.com/rd20karim/M2T-Segmentation).

**Synchronous captioning** aims to generate text aligned with the time evolution of 3D human motion. Implicitly, this mapping provides fine-grained action recognition and unsupervised event localization with temporal phrase grounding through unsupervised motion-language segmentation.



## Bibtex
If you find this work useful in your research, please cite:
```
@article{radouane2024ControlledTransformer,
      title={Transformer with Controlled Attention for Synchronous Motion Captioning}, 
      author={Karim Radouane and Sylvie Ranwez and Julien Lagarde and Andon Tchechmedjiev},
      booktitle = {Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-26)},
      year      = {2026}
}
```

## Quick start

```
conda env create -f environment.yaml
conda activate wbpy310
python -m spacy download en-core-web-sm
```
You need also to install wandb for hyperparameters tuning: 
``
pip install wandb
``
## Preprocess datasets

For both HumanML3D and KIT-MLD (augmented versions) you can follow the steps here: [project link](https://github.com/rd20karim/M2T-Segmentation?tab=readme-ov-file#preprocess-dataset)


## Training and evaluation

**Training:** Using [train.py](train.py) script allows you to train the model by defining the config file and the dataset path. You can also use the wandb for hyperparameters tuning.

**Evaluation:** Using [evaluate_transformer.py](evaluate_transformer.py) script, you can evaluate the model on the test set by defining the config file and the model checkpoint.


Note: More details and Models checkpoints will be available soon.

## Demonstration

In the following visual animations, we present the synchronized output results for some motions, mainly compositional, which include samples containing two or more actions:


<div align="center">
  <img src="synch_mesh_gifs/getsOnFours_crawls.gif" width="250" height="250">
  <img src="synch_mesh_gifs/picks_sits.gif"  width="250" height="250">
  <img src="synch_mesh_gifs/walks_turns_cartwheel.gif"  width="250" height="250">
</div>



<div align="center">
  <img src="synch_mesh_gifs/bends_raisesArms.gif" width="250" height="250">
  <img src="synch_mesh_gifs/walks_wipes.gif"  width="250" height="250">
  <img src="synch_mesh_gifs/kicks_left.gif"  width="250" height="250">
</div>


<div align="center">
  <img src="synch_mesh_gifs/jogs_forward_stops.gif" width="250" height="250">
  <img src="synch_mesh_gifs/sitting_stands.gif"  width="250" height="250">
  <img src="synch_mesh_gifs/walks_up_ladder.gif"  width="250" height="250">
</div>


## Architecture

Our method introduces mechanisms to **control self- and cross-attention** distributions of the Transformer, allowing interpretability and time-aligned text generation. 
We achieve this through **masking strategies** and **structuring losses** that push the model to maximize attention only on the most important frames contributing to the generation of a motion word. 
These constraints aim to **prevent undesired mixing of information** in attention maps and to provide a monotonic attention distribution across tokens. 
Thus, the cross attentions of tokens are used for **progressive text generation** in synchronization with human motion sequences.

<div align="center">
    <img src="Concept_Transformer_Synch.png" width=500">
</div>


## Motion Frozen in Time

* Phrase-level

<div align="center">
    <img src="Frozen/bends_raisearms.PNG" width=500">
    <img src="Frozen/walks_picks_walksback.PNG" width=500">
</div>


* Word-level 

<div align="center">
    <img src="Frozen/frozen_all_3622.png" width=500">
    <img src="Frozen/frozen_all_4015.png" width=500">
</div>





