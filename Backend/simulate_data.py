### Backend for simulating data

## Import packages:

import sys, os, os.path, h5py, time   #pandas, numpy, pool, diffpy, time
import pandas as pd
import numpy as np

from os import walk
from multiprocessing import Pool
from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from diffpy.Structure import loadStructure, Structure, Lattice, Atom

from diffpy.Structure import Structure   #DiffPy imports
from diffpy.Structure import Atom
from diffpy.Structure import loadStructure, Structure, Lattice

from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.cluster.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.lattice.hexagonal import Hexagonal, HexagonalClosedPacked, Graphite
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron

from tqdm import tqdm
import shutil, mendeleev
from shutil import copyfile
from scipy import spatial

## Define functions:

def sort_filenames(xyz_path, print_level):
    #Sort through all filenames and sort them into different lists based on structure:
    _, _, filenames = next(walk(xyz_path)) #loads all filenames into a list called filenames
    #structures_list = ["SC", "FCC", "BCC", "HCP", "Icosahedron", "Decahedron", "Octahedron"]
    SC_filenames, FCC_filenames, BCC_filenames, HCP_filenames, ICO_filenames, DEC_filenames, OCT_filenames = [], [], [], [], [], [], []
    for xyzfile in filenames:
        if xyzfile[0] == "S":
            SC_filenames.append(xyzfile)
        if xyzfile[0] == "F":
            FCC_filenames.append(xyzfile)
        if xyzfile[0] == "B":
            BCC_filenames.append(xyzfile)
        if xyzfile[0] == "H":
            HCP_filenames.append(xyzfile)
        if xyzfile[0] == "I":
            ICO_filenames.append(xyzfile)
        if xyzfile[0] == "D":
            DEC_filenames.append(xyzfile)
        if xyzfile[0] == "O":
            OCT_filenames.append(xyzfile)
    sorted_filenames = [SC_filenames, FCC_filenames, BCC_filenames, HCP_filenames, ICO_filenames, DEC_filenames, OCT_filenames]
    if print_level == True:
        print("All xyz files have been sorted into lists")
    return sorted_filenames

def get_simulate_data_backend():
	#Sorted filenames:
	xyz_path = "xyz_files/natoms200/" #Path where the xyz files are
	sorted_filenames = sort_filenames(xyz_path, print_level = False)
	sorted_filenames_flat = [item for sublist in sorted_filenames for item in sublist]
	
	#Instantiate the PDF_generator class
	PDF_obj = PDF_generator() #This doesn't even need to be here lol

	return PDF_obj#, sorted_filenames_flat, xyz_path

def make_df_parallelized(sorted_filenames_flat, train_multip, valtest_multip, n_processes, xyz_path):
    
    sorted_filenames = sort_filenames(xyz_path, print_level = False)
    sorted_filenames_flat = [item for sublist in sorted_filenames for item in sublist]

    #Instantiate the PDF_generator class
    PDF_obj = PDF_generator() 

    t1 = time.time()
    
    print(f"Making PDF dataset from the clusters in {xyz_path}.\nThe training set will have {train_multip} PDFs from each cluster and the validation and test set will have {valtest_multip} PDFs from each cluster\nMultithreading: {n_processes} processes")
    
    reset_parameters() #Måske en god idé lige at få lagt den her en enkelt gang
    PDF_train, PDF_validation, PDF_test  = [], [], []

    xyzfileindexes = sorted([n for n in range(len(sorted_filenames_flat))]*(train_multip+2*valtest_multip))

    p = Pool(n_processes)

    result = p.map(PDF_from_index, xyzfileindexes)

    p.close()
    p.join()

    PDF_train = []
    PDF_validation = []
    PDF_test = []

    n = 0
    for _ in range(int(len(result)/(train_multip+2*valtest_multip))):
        for i in range(train_multip):
            PDF_train.append(result[n])
            n = n + 1
        for j in range(valtest_multip):
            PDF_validation.append(result[n])
            n = n + 1
            PDF_test.append(result[n])
            n = n + 1

    index_list = np.arange(1, 301, 1)
    index_list = np.append(index_list, "qmin")
    index_list = np.append(index_list, "qmax")
    index_list = np.append(index_list, "qdamp")
    #index_list = np.append(index_list, "biso")
    index_list = np.append(index_list, "filename")

    df_train = pd.DataFrame(data = PDF_train, columns = index_list)
    df_val = pd.DataFrame(data = PDF_validation, columns = index_list)
    df_test = pd.DataFrame(data = PDF_test, columns = index_list)

    #Reduce Memory
    df_train.iloc[:,1:-1] = reduce_mem_usage(df_train.iloc[:,1:-1])
    df_val.iloc[:,1:-1] = reduce_mem_usage(df_val.iloc[:,1:-1])
    df_test.iloc[:,1:-1] = reduce_mem_usage(df_test.iloc[:,1:-1])

    #Split train og val i X og y
    X_train,y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_val,y_val = df_val.iloc[:,0:-1], df_val.iloc[:,-1]
    X_test,y_test = df_test.iloc[:,0:-1], df_test.iloc[:,-1]
    print(f"df_train dimensions: {len(PDF_train)} rows X {len(PDF_train[0])} columns")
    print(f"df_val dimensions: {len(PDF_validation)} rows X {len(PDF_validation[0])} columns")
    print(f"df_test dimensions: {len(PDF_test)} rows X {len(PDF_test[0])} columns")
    t2 = time.time()
    print("Time spent on making PDFs and preparing dataframes:", str((t2-t1)/60)[0:4], " min")
    return X_train, y_train, X_val, y_val, X_test, y_test #, (time.time()-start_time)/60

def save_PDFs(folder_name, X_train, y_train, X_val, y_val, X_test, y_test):
  
    pdf_path = "PDF_datasets/"
    if os.path.isdir(pdf_path + folder_name):
        print("This directory already exists. Choose a different folder name in order to not overwrite existing dataset...")
        return
    else:
        os.makedirs(pdf_path + folder_name)
    X_train.to_hdf(pdf_path + folder_name + "/X_train.h5", key='df', mode='w')
    y_train.to_hdf(pdf_path + folder_name + "/y_train.h5", key='df', mode='w')
    X_val.to_hdf  (pdf_path + folder_name + "/X_val.h5", key='df', mode='w')
    y_val.to_hdf  (pdf_path + folder_name + "/y_val.h5", key='df', mode='w')
    X_test.to_hdf (pdf_path + folder_name + "/X_test.h5", key='df', mode='w')
    y_test.to_hdf (pdf_path + folder_name + "/y_test.h5", key='df', mode='w')
    print("PDF dataset saved to " + pdf_path + folder_name + "/")

def reduce_mem_usage(df): #tjek bed den her om at holde kæft evt.
    """ 
    iterate through all the columns of a dataframe and 
    modify the data type to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    #print(('Memory usage of dataframe is {:.2f}' 
    #                 'MB').format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max <\
                  np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max <\
                   np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max <\
                   np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max <\
                   np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max <\
                   np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max <\
                   np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    #print(('Memory usage after optimization is: {:.2f}' 
    #                          'MB').format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) 
    #                                         / start_mem))
    
    return df

def reset_parameters():
    PDF_obj.qmin=0.7
    PDF_obj.qmax=25
    PDF_obj.qdamp=0.04
    PDF_obj.rmin=0
    PDF_obj.rmax=30
    PDF_obj.rstep=0.1
    PDF_obj.biso=1
    PDF_obj.delta2=2

def vary_parameters():
    PDF_obj.qmin = np.random.uniform(0, 2)
    PDF_obj.qmax = np.random.uniform(12, 25)
    PDF_obj.qdamp = np.random.uniform(0.01, 0.04)
    PDF_obj.biso = np.random.uniform(0.1, 1)    

def PDF_from_index_test(xyzfileindex):
    print(xyz_path)
    return None

def PDF_from_index(xyzfileindex):

    #xyz_path = 'xyz_files/natoms200/'

    vary_parameters()
    xgrid, xyz_pdf = PDF_obj.genPDF(xyz_path+sorted_filenames_flat[xyzfileindex], fmt='xyz') #PDFs are simulated here
    xyz_pdf = xyz_pdf/max(xyz_pdf) #Normalisation
    xyz_pdf = np.append(xyz_pdf, np.array([PDF_obj.qmin, PDF_obj.qmax, PDF_obj.qdamp])) #Adds the parameters Qmin, Qmax and Qdamp
    xyz_pdf = np.append(xyz_pdf, int(xyzfileindex)) #Adds the xyz file index at the end of the PDF
    return xyz_pdf

class PDF_generator:
    def __init__(self, qmin=0.5, qmax=30, qdamp=0.03, rmin=0, rmax=30, rstep=0.1, biso=0.3, delta2=0):
        """
        Class for simulating total scattering data for both crystal and cluster files.

        Parameters
        ----------
        qmin : Smallest q-vector included (float)
        qmax : Instrumental resolution (float)
        qdamp : Instrumental dampening (float)
        rmin : Start of r-grid (float)
        rmax : End of r-grid (float)
        rstep : Spacing between r-grid (float)
        biso : Vibration (float (float)
        delta2 : Correlated vibration (float)
        """

        self.qmin = qmin
        self.qmax = qmax  # Instrument resolution
        self.qdamp = qdamp  # Instrumental dampening
        self.rmin = rmin  # Smallest r value
        self.rmax = rmax  # Can not be less than 10 AA
        self.rstep = rstep  # Nyquist for qmax = 30
        self.biso = biso  # Atomic vibration
        self.delta2 = delta2  # Corelated vibration


    def genPDF(self, clusterFile, fmt='cif'):
        """
        Simulates PDF for input structure.

        Parameters
        ----------
        clusterFile : path for input file (str)
        fmt : cif or xyz, depending on structure type (str)

        Returns
        -------
        r : r-grid (NumPy array)
        Gr : Intensities (NumPy array)
        """
        stru = loadStructure(clusterFile)

        stru.B11 = self.biso
        stru.B22 = self.biso
        stru.B33 = self.biso
        stru.B12 = 0
        stru.B13 = 0
        stru.B23 = 0

        if fmt.lower() == 'cif':
            PDFcalc = PDFCalculator(rmin=self.rmin, rmax=self.rmax, rstep=self.rstep,
                                    qmin=self.qmin, qmax=self.qmax, qdamp=self.qdamp, delta2=self.delta2)
        elif fmt.lower() == 'xyz':
            PDFcalc = DebyePDFCalculator(rmin=self.rmin, rmax=self.rmax, rstep=self.rstep,
                                         qmin=self.qmin, qmax=self.qmax, qdamp=self.qdamp, delta2=self.delta2)
        r0, g0 = PDFcalc(stru)

        self.r = np.array(r0)
        self.Gr = np.array(g0)

        return self.r, self.Gr

    def __str__(self):
        return 'PDF parameters:\n\tqmin: {}\n\tqmax: {}\n\tqdamp: {}\n\trmin: {}' \
                              '\n\trmax {}\n\trstep: {}\n\tbiso: {} \n\tDelta2: {}\n'\
                                        .format(self.qmin, self.qmax, self.qdamp, self.rmin,
                                                self.rmax, self.rstep, self.biso, self.delta2)

structures_list = ["SC", "FCC", "BCC", "HCP", "Icosahedron", "Decahedron", "Octahedron"]

def make_data(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
    if Geometry=="SC" or Geometry=="FCC" or Geometry=="BCC":
        make_data_SC_FCC_BCC(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
    if Geometry=="HCP":
        make_data_HCP(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
    if Geometry=="Icosahedron":
        make_data_Icosahedron(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
    if Geometry=="Decahedron":
        make_data_Decahedron(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
    if Geometry=="Octahedron":
        make_data_Octahedron(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
    return None

def make_data_SC_FCC_BCC(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
    global h, k, l, atom, lc
    if numberatoms < 101:
        h_list = [1, 2, 3, 4, 5, 6, 7, 8]
        k_list = [1, 2, 3, 4, 5, 6, 7]
        l_list = [1, 2, 3, 4, 5, 6, 7]
    elif numberatoms < 201 and numberatoms > 101:
        h_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        l_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif numberatoms > 201:
        h_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        l_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    possible_structures = len(h_list)*len(k_list)*len(l_list)*len(atom_list)*numberOfBondLengths
    pbar = tqdm(total=possible_structures)
    for atom in atom_list:
        structure_list = []
        for h in h_list:
            for k in k_list:
                for l in l_list:
                    if h <= k and h <= l and k <= l:
                        #atom_cov_radii = covalent_radii[atomic_numbers[atom]]
                        atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
                        lc_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
                        if numberOfBondLengths == 1:
                            lc_list = [atom_cov_radii*2]
                        for lc in lc_list:
                            stru1, structure_type = structure_maker(Geometry)
                            xyz1 = stru1.get_positions()
                            if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
                                cluster = Structure([Atom(atom, xi) for xi in xyz1])
                                if new_structure_checker(xyz1, structure_list) == False:
                                    structure = np.zeros((numberatoms, 4))
                                    for i in range(len(xyz1)):
                                        structure[i, 0] = atomic_numbers[atom]
                                        structure[i, 1:] = xyz1[i]
                                    structure_list.append(xyz1)
                                    cluster.write(path+str(Geometry)+"_h_"+str(h)+"_k_"+str(k)+"_l_"+str(l)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", format="xyz")
                                    #generator = simPDFs(path+str(Geometry)+"_h_"+str(h)+"_k_"+str(k)+"_l_"+str(l)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", "./")
                                    #r, Gr = generator.getPDF()
                                    #y = [structure_type, h, k, l, 0, 0, 0, 0, 0, 0, 0, 0, atomic_numbers[atom], lc, 0]+Gr.tolist()
                                    #np.savetxt(path+str(Geometry)+"_h_"+str(h)+"_k_"+str(k)+"_l_"+str(l)+"_atom_"+str(atom)+"_lc_"+str(lc)+"_LABEL.txt", y)
                    pbar.update(1)
    pbar.close()     
    return None

def make_data_HCP(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
    global size1, size2, size3, atom, lc1, lc2
    if numberatoms < 101:
        size1_list = [1, 2, 3, 4, 5, 6, 7, 8]
        size2_list = [1, 2, 3, 4, 5, 6, 7, 8]
        size3_list = [1, 2, 3, 4, 5, 6, 7, 8]
    elif numberatoms < 201 and numberatoms > 101:
        size1_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        size2_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        size3_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif numberatoms > 201:
        size1_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        size2_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        size3_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    possible_structures = len(size1_list)*len(size2_list)*len(size3_list)*len(atom_list)*numberOfBondLengths
    pbar = tqdm(total=possible_structures)
    for atom in atom_list:
        structure_list = []
        for size1 in size1_list:
            for size2 in size2_list:
                for size3 in size3_list:
                    if size1 <= size2 and size1 <= size3 and size2 <= size3:	
                        #atom_cov_radii = covalent_radii[atomic_numbers[atom]]
                        atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
                        lc1_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
                        if numberOfBondLengths == 1:
                            lc1_list = [atom_cov_radii*2]
                        for lc1 in lc1_list:
                            #lc2_list = np.linspace(lc1*2*0.99 , lc1*2*1.01, 10)*1.633
                            lc2_list = np.asarray([lc1])*1.633
                            for lc2 in lc2_list:
                                stru1, structure_type = structure_maker(Geometry)
                                xyz1 = stru1.get_positions()
                                if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
                                    cluster = Structure([Atom(atom, xi) for xi in xyz1])
                                    if new_structure_checker(xyz1, structure_list) == False:
                                        structure = np.zeros((numberatoms, 4))
                                        for i in range(len(xyz1)):
                                            structure[i, 0] = atomic_numbers[atom]
                                            structure[i, 1:] = xyz1[i]
                                        structure_list.append(xyz1)
                                        cluster.write(path+str(Geometry)+"_Size1_"+str(size1)+"_Size2_"+str(size2)+"_Size3_"+str(size3)+"_atom_"+str(atom)+"_lc1_"+str(lc1)+"_lc2_"+str(lc2)+".xyz", format="xyz")
                                        #generator = simPDFs(path+str(Geometry)+"_Size1_"+str(size1)+"_Size2_"+str(size2)+"_Size3_"+str(size3)+"_atom_"+str(atom)+"_lc1_"+str(lc1)+"_lc2_"+str(lc2)+".xyz", "./")
                                        #r, Gr = generator.getPDF()
                                        #y = [structure_type, 0, 0, 0, size1, size2, size3, 0, 0, 0, 0, 0, atomic_numbers[atom], lc1, lc2]+Gr.tolist()
                                        #np.savetxt(path+str(Geometry)+"_Size1_"+str(size1)+"_Size2_"+str(size2)+"_Size3_"+str(size3)+"_atom_"+str(atom)+"_lc1_"+str(lc1)+"_lc2_"+str(lc2)+"_LABEL.txt", y)
                                pbar.update(1)
    pbar.close()
    return None

def make_data_Icosahedron(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
    global atom, shell, lc
    if numberatoms < 101:
        shell_list = [1, 2, 3, 4]
    elif numberatoms < 201 and numberatoms > 101:
        shell_list = [1, 2, 3, 4, 5]
    elif numberatoms > 201:
        shell_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    possible_structures = len(shell_list)*len(atom_list)*numberOfBondLengths
    pbar = tqdm(total=possible_structures)
    for atom in atom_list:
        structure_list = []
        for shell in shell_list:	
            #atom_cov_radii = covalent_radii[atomic_numbers[atom]]
            atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
            lc_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
            if numberOfBondLengths == 1:
                lc_list = [atom_cov_radii*2]
            for lc in lc_list:
                stru1, structure_type = structure_maker(Geometry)
                xyz1 = stru1.get_positions()
                if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
                    cluster = Structure([Atom(atom, xi) for xi in xyz1])
                    if new_structure_checker(xyz1, structure_list) == False:
                        structure = np.zeros((numberatoms, 4))
                        for i in range(len(xyz1)):
                            structure[i, 0] = atomic_numbers[atom]
                            structure[i, 1:] = xyz1[i]
                        structure_list.append(xyz1)
                        cluster.write(path+str(Geometry)+"_shell_"+str(shell)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", format="xyz")
                        #generator = simPDFs(path+str(Geometry)+"_shell_"+str(shell)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", "./")
                        #r, Gr = generator.getPDF()
                        #y = [structure_type, 0, 0, 0, 0, 0, 0, shell, 0, 0, 0, 0, atomic_numbers[atom], lc, 0]+Gr.tolist()
                        #np.savetxt(path+str(Geometry)+"_shell_"+str(shell)+"_atom_"+str(atom)+"_lc_"+str(lc)+"_LABEL.txt", y)
                pbar.update(1)
    pbar.close()
    return None

def make_data_Decahedron(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
    global atom, p, q, r, lc
    if numberatoms < 101:
        p_list = [1, 2, 3, 4]
        q_list = [1, 2, 3, 4]
        r_list = [0, 1, 2, 3, 4]
    elif numberatoms < 201 and numberatoms > 101:
        p_list = [1, 2, 3, 4, 5]
        q_list = [1, 2, 3, 4, 5]
        r_list = [0, 1, 2, 3, 4, 5]
    elif numberatoms > 201:
        p_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        q_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        r_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    possible_structures = len(p_list)*len(q_list)*len(r_list)*len(atom_list)*numberOfBondLengths
    pbar = tqdm(total=possible_structures)
    for atom in atom_list:
        structure_list = []
        for p in p_list:
            for q in q_list:
                for r in r_list:	
                    #if p <= q and p <= r and q <= r:	
                    #atom_cov_radii = covalent_radii[atomic_numbers[atom]]
                    atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
                    lc_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
                    if numberOfBondLengths == 1:
                        lc_list = [atom_cov_radii*2]
                    for lc in lc_list:
                        stru1, structure_type = structure_maker(Geometry)
                        xyz1 = stru1.get_positions()
                        if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
                            cluster = Structure([Atom(atom, xi) for xi in xyz1])
                            if new_structure_checker(xyz1, structure_list) == False:
                                structure = np.zeros((numberatoms, 4))
                                for i in range(len(xyz1)):
                                    structure[i, 0] = atomic_numbers[atom]
                                    structure[i, 1:] = xyz1[i]
                                structure_list.append(xyz1)
                                cluster.write(path+str(Geometry)+"_p_"+str(p)+"_q_"+str(q)+"_r_"+str(r)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", format="xyz")
                                #generator = simPDFs(path+str(Geometry)+"_p_"+str(p)+"_q_"+str(q)+"_r_"+str(r)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", "./")
                                #r_PDF, Gr = generator.getPDF()
                                #y = [structure_type, 0, 0, 0, 0, 0, 0, 0, p, q, r, 0, atomic_numbers[atom], lc, 0]+Gr.tolist()
                                #np.savetxt(path+str(Geometry)+"_p_"+str(p)+"_q_"+str(q)+"_r_"+str(r)+"_atom_"+str(atom)+"_lc_"+str(lc)+"_LABEL.txt", y)
                        pbar.update(1)
    pbar.close()
    return None

def make_data_Octahedron(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
    global atom, length, lc
    if numberatoms < 101:
        length_list = [2, 3, 4, 5]
    elif numberatoms < 201 and numberatoms > 101:
        length_list = [2, 3, 4, 5, 6, 7]
    elif numberatoms > 201:
        length_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    possible_structures = len(length_list)*len(atom_list)*numberOfBondLengths
    pbar = tqdm(total=possible_structures)
    for atom in atom_list:
        structure_list = []
        for length in length_list:	
            #atom_cov_radii = covalent_radii[atomic_numbers[atom]]
            atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
            lc_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
            if numberOfBondLengths == 1:
                lc_list = [atom_cov_radii*2]
            for lc in lc_list:
                stru1, structure_type = structure_maker(Geometry)
                xyz1 = stru1.get_positions()
                if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
                    cluster = Structure([Atom(atom, xi) for xi in xyz1])
                    if new_structure_checker(xyz1, structure_list) == False:
                        structure = np.zeros((numberatoms, 4))
                        for i in range(len(xyz1)):
                            structure[i, 0] = atomic_numbers[atom]
                            structure[i, 1:] = xyz1[i]
                        structure_list.append(xyz1)
                        cluster.write(path+str(Geometry)+"_length_"+str(length)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", format="xyz")
                        #generator = simPDFs(path+str(Geometry)+"_length_"+str(length)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", "./")
                        #r, Gr = generator.getPDF()
                        #y = [structure_type, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, length, atomic_numbers[atom], lc, 0]+Gr.tolist()
                        #np.savetxt(path+str(Geometry)+"_length_"+str(length)+"_atom_"+str(atom)+"_lc_"+str(lc)+"_LABEL.txt", y)
                pbar.update(1)
    pbar.close()
    return None

def new_structure_checker(Input_array, list_search):
    for i in range(len(list_search)):
        if np.all(Input_array == list_search[i]) == True:
            return True
        else:
            pass
    return False

def remove_same_structures(path, savePath): #Possibly not used in any other funcs
					    
    files = os.listdir(path)
    softadj_list = []
    remove_files = []
    pbar = tqdm(total=len(files))

    for file in files:
        structure = loadStructure(data_dir+file)
        softadj = np.zeros((len(structure), len(structure)))
        for i in range(len(structure)):
            for j in range(len(structure)):
                atom1 = np.array([structure.x[i], structure.y[i], structure.z[i]])
                atom2 = np.array([structure.x[j], structure.y[j], structure.z[j]])
                dist = np.linalg.norm(atom1-atom2)
                softadj[i][j] = dist
        if new_structure_checker(softadj, softadj_list):
            remove_files.append(file)
        else:
            softadj_list.append(softadj)
            copyfile(path+file, savePath+file)
        pbar.update(1)
    pbar.close()

    return remove_files

def structure_maker(Geometry):
    global h, k, l, atom, lc, lc1, lc2, size1, size2, size3, shell, p, q, r, length
    if Geometry=="SC": 
        stru = SimpleCubic(atom, surfaces=[[1,0,0], [1,1,0], [1,1,1]], layers=(h,k,l), latticeconstant=lc)
        structure_type = 0
    if Geometry=="FCC": 
        stru = FaceCenteredCubic(atom, surfaces=[[1,0,0], [1,1,0], [1,1,1]], layers=(h,k,l), latticeconstant=2*np.sqrt(0.5*lc**2))
        structure_type = 1
    if Geometry=="BCC": 
        stru = BodyCenteredCubic(atom, surfaces=[[1,0,0], [1,1,0], [1,1,1]], layers=(h,k,l), latticeconstant=lc)
        structure_type = 2
    if Geometry=="HCP": 
        stru = HexagonalClosedPacked(symbol=atom, latticeconstant=(lc1, lc2), size=(size1, size2, size3))
        structure_type = 3
    if Geometry=="Icosahedron": 
        stru = Icosahedron(atom, shell, latticeconstant=2*np.sqrt(0.5*lc**2))
        structure_type = 4
    if Geometry=="Decahedron": 
        stru = Decahedron(atom, p, q, r, latticeconstant=2*np.sqrt(0.5*lc**2))
        structure_type = 5
    if Geometry=="Octahedron": 
        stru = Octahedron(atom, length, latticeconstant=2*np.sqrt(0.5*lc**2))
        structure_type = 6
    return stru, structure_type

def remove_dub(dir1):
    files1 = sorted(os.listdir(dir1))
    remove_list = []

    pbar = tqdm(total=len(files1))
    for idx1 in range(len(files1)):
        file1 = files1[idx1]
        dat1 = np.loadtxt(dir1+'/'+file1, skiprows=1, dtype=np.str)
        dat1 = np.array(dat1.T[1:], dtype=np.float)
        dist_mat1 = spatial.distance.cdist(dat1.T, dat1.T, 'euclidean')
        dist_mat1 = np.sort(dist_mat1.reshape(-1))
        pbar.update()
        for idx2 in range(idx1, len(files1)):
            file2 = files1[idx2]

            if file1 == file2:
                continue
            elif file2 in remove_list or file1 in remove_list:
                continue

            dat2 = np.loadtxt(dir1+'/'+file2, skiprows=1, dtype=np.str)
            dat2 = np.array(dat2.T[1:], dtype=np.float)

            if dat1.shape == dat2.shape:
                dist_mat2 = spatial.distance.cdist(dat2.T, dat2.T, 'euclidean')
                dist_mat2 = np.sort(dist_mat2.reshape(-1))

                val = np.array_equal(dist_mat1, dist_mat2)

                if val == True:
                    #print('\t\t',file1, file2) #
                    remove_list.append(file1)
                    os.remove(dir1+'/'+file1)
                    #print(file1, 'has been deleted\n') #
    return None

def simulate_clusters(folder_name, maximum_atoms, minimum_atoms):
    data_dir = "xyz_files/" + folder_name + "/"
    start_time = time.time()
    
    if os.path.isdir(data_dir):
        print("This directory already exists. Choose a different folder name...")
        return
    os.makedirs(data_dir)
    structure_type_list = ["SC", "BCC", "FCC", "HCP", "Icosahedron", "Decahedron", "Octahedron"]
    numberOfBondLengths=1
    
    print("Simulating metal nanoparticle clusters from " + str(minimum_atoms) + " to " + str(maximum_atoms) + " atoms of structure types: SC, BCC, FCC, HCP, Icosahedron, Decahedron and Octahedron")

    for structure_type in structure_type_list:
        make_data(data_dir, maximum_atoms, structure_type, minimum_atoms=5, numberOfBondLengths=1)

    print("All clusters simulated and saved in folder: " + str(data_dir))
    print("Now removing all duplicates. This is a time-consuming process. Lean back and crack open a cold one")

    remove_dub(data_dir)

    print("All duplicates removed")
    end_time = time.time()
    print("Cluster simulation complete! " + str(len(os.listdir(data_dir))) + " metal clusters created. It took " + str((end_time - start_time)/60)[0:5] + " minutes.")

global atom_list #sets atom_list to be a global variable, e.g. also defined inside defined functions
atom_list = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
             'Y', 'Zr', 'Nb', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
             'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']