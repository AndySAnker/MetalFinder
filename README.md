[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6221f17357a9d20c9a729ecb)  |  [Paper](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00086e)

# MetalFinder

MetalFinder is a tree-based supervised learning algorithm which can predict the mono-metallic nanoparticle (MMNP) from a Pair Distribution Function (PDF).

![alt text](images/MetalFinder.png "MetalFinder")

Currently MetalFinder is limited to MMNPs with up to 200 atoms of the 7 different structure types: 
Cubic (sc), body-centered cubic (bcc), face-centered cubic (fcc), hexagonal closed packed (hcp), decahedral, icosahedral and octahedral

1. [MetalFinder](#metalfinder)
2. [Getting started (own computer)](#getting-started-own-computer)
    1. [Install requirements](#install-requirements)
    2. [Simulate data](#simulate-data)
    3. [Train model](#train-model)
    4. [Predict](#predict)
3. [Author](#author)
4. [Cite](#cite)
5. [Acknowledgments](#Acknowledgments)
6. [License](#license)  


# Getting started (own computer)
Follow these step if you want to train MetalFinder and predict with MetalFinder locally on your own computer.

## Install requirements
See the [install](/install) folder. 

## Simulate data
To simulate the data used to train MetalFinder open the file:
```
jupyter notebook 1_Simulate_Data.ipynb
```
Follow the instructions in the 1_Simulate_Data.ipynb file.

To download the xyz-files (atomic models) and PDF dataset instead of simulating both locally, see the [data](/data) folder.

 

## Train model
To train your own MetalFinder model simply run:
```
jupyter notebook 2_Training.ipynb
```
Follow the instructions in the 2_Training.ipynb file. To download the model used for prediction in the article, instead of training at your own computer, see the [data](/data) folder.

## Predict
To predict a MMNP using MetalFinder or your own model on a PDF:
```
jupyter notebook 3_Testing.ipynb
```
Follow the instructions in the 3_Testing.ipynb file.


# Authors
__Andy S. Anker__<sup>1</sup>   
__Emil T. S. Kjær__<sup>1</sup>  
__Marcus N. Weng__<sup>1</sup>  
__Simon J. L. Billinge__<sup>2, 3</sup>     
__Raghavendra Selvan__<sup>4, 5</sup>  
__Kirsten M. Ø. Jensen__<sup>1</sup>    
 
<sup>1</sup> Department of Chemistry and Nano-Science Center, University of Copenhagen, 2100 Copenhagen Ø, Denmark.   
<sup>2</sup> Department of Applied Physics and Applied Mathematics Science, Columbia University, New York, NY 10027, USA.   
<sup>3</sup> Condensed Matter Physics and Materials Science Department, Brookhaven National Laboratory, Upton, NY 11973, USA.    
<sup>4</sup> Department of Computer Science, University of Copenhagen, 2100 Copenhagen Ø, Denmark.   
<sup>5</sup> Department of Neuroscience, University of Copenhagen, 2200, Copenhagen N.    

Should there be any question, desired improvement or bugs please contact us on GitHub or 
through email: __ansoan@dtu.dk__ or __etsk@chem.ku.dk__.

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!
```
@article{kjær2022DeepStruc,
title={DeepStruc: Towards structure solution from pair distribution function data using deep generative models},
author={Emil T. S. Kjær, Andy S. Anker, Marcus N. Weng, Simon J. L. Billinge, Raghavendra Selvan, Kirsten M. Ø. Jensen},
year={2022}}
```

# License
This project is licensed under the Apache License Version 2.0, January 2004 - see the [LICENSE](LICENSE) file for details.
