[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6221f17357a9d20c9a729ecb)  |  [Paper](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00086e)

# Data introduction
This readme contains instructions to generate the data used in this project and download links to a zip folder with the simulated atomic models (xyz_files) and the simulated PDF dataset

1. [Generate data](#generate-data)
2. [Download links](#download-links)

# Generate data
[DiffPy-CMI](https://www.diffpy.org/products/diffpycmi/index.html) in required to simulate PDFs, which only runs on Linux or macOS. To run it on a Windows computer
please use the [Ubuntu subsystem](https://ubuntu.com/tutorials/ubuntu-on-windows#1-overview).

The Metalfinder script is divided into three parts (Simulating data, training model, testing model). The first notebook (1_Simulating_data.ipynb) is used to simulate atomic models of metallic nanomarticles and from those simulate PDFs with varied parameters. In the article, atomic models are simulated in seven structure types (sc, bcc, fcc, hcp, Decahedron, Octahedron, Icosahedron) from 6 to 200 atoms. The PDF dataset is simulated with varied parameters with 100 PDFs per xyz-file in the training set, and 15 PDFs per xyz-file in the validation set as well as 15 PDFs per xyz-file in the test set.

# [Download link](https://zenodo.org/record/6418055#.Yk2WIJNBzDI)
The following can be downloaded from the download link: https://zenodo.org/record/6418055#.Yk2WIJNBzDI
- The xyz-files from 6 to 200 atoms
- The PDF dataset simulated with seed 37
- The tree-based model trained with seed 37 and 100 PDFs/xyz-file in the training set and 15 PDFs/xyz-file in the validation set and the test set respectively
