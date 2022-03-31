[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6221f17357a9d20c9a729ecb)  |  [Paper] XXX

# Install
To run MetalFinder you will need some packages, which are dependent on your computers specifications. 

All  packages can be install through the requirement files or the install.sh file. 
First go to your desired conda environment.
 ```
conda activate <env_name>
``` 
Or create one and activate it afterwards.
```
conda create --name env_name python=3.7
``` 

Now install the required packages through the requirement files.
DiffPy-CMI might be necessary for some functions of MetalFInder(see how to [HERE](https://www.diffpy.org/products/diffpycmi/index.html)).
### Simulate Data
Install through the install.sh file.
```
sh install_bash_files/install_simulate.sh
``` 
### Training
Install through the install.sh file.
```
sh install_bash_files/install_training.sh
``` 

### Testing
Install through the install.sh file.
```
sh install_bash_files/install_testing.sh
``` 

