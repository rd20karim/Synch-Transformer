## Description

Official implementation of Synch-Transformer for synchronous motion captioning:

<div align="center">

[<span style="font-size: 25px;"> <span style="color:darkviolet; font-weight:darkbold; font-size: 30px">Transformer with Controlled Attention for Synchronous Captioning </span>]()

[![arxiv](https://img.shields.io/badge/arXiv-Synch_Transformer-cyan?logo=arxiv)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

This work introduces a Transformer-based design to address the task of motion-to-text synchronization, as introduced in this previous project [m2t-segmentation](https://github.com/rd20karim/M2T-Segmentation).

**Synchronous captioning** aims to generate text aligned with the time evolution of 3D human motion. Implicitly, this mapping provides fine-grained action recognition and unsupervised event localization with temporal phrase grounding through unsupervised motion-language segmentation.

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


