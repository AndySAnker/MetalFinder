### Backend for testing

###Import packages 
import sys, os, os.path, h5py, time, shutil

from os import walk
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from shutil import copy

from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize.minpack import leastsq
from diffpy.Structure import loadStructure
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.fitbase import FitRecipe, FitResults

## From new fitting
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize.minpack import leastsq
#from diffpy.Structure import loadStructure
from diffpy.Structure import Structure
from diffpy.Structure import Atom
from diffpy.Structure.expansion import supercell
from diffpy.srfit.pdf import PDFContribution, DebyePDFGenerator, PDFParser
from diffpy.srfit.fitbase import FitRecipe, FitResults,FitContribution
from ase.cluster.decahedron import Decahedron
from ase.io import write
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.cluster.wulff import wulff_construction
import time, ase, glob, pdb
from ase.cluster.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.lattice.cubic import Diamond
from ase.lattice.tetragonal import SimpleTetragonal, CenteredTetragonal
from ase.lattice.orthorhombic import SimpleOrthorhombic, BaseCenteredOrthorhombic, FaceCenteredOrthorhombic, BodyCenteredOrthorhombic
from ase.lattice.monoclinic import SimpleMonoclinic, BaseCenteredMonoclinic
from ase.lattice.triclinic import Triclinic
from ase.lattice.hexagonal import Hexagonal, HexagonalClosedPacked, Graphite
from diffpy.srfit.fitbase import Profile #tilføjet til backend
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from scipy.optimize.minpack import leastsq
from scipy.optimize import minimize
#import pandas as pd
from multiprocessing import Pool

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

def get_testing_backend(xyz_folder_name):
	#Sorted filenames:
	xyz_path = "xyz_files/" + xyz_folder_name + "/" #Path where the xyz files are
	sorted_filenames = sort_filenames(xyz_path, print_level = False)
	sorted_filenames_flat = [item for sublist in sorted_filenames for item in sublist]

	return sorted_filenames_flat, xyz_path #Eller bare lav den her global? Det giver et simplere dokument måske idk

def load_PDFs(folder_name):
    pdf_path = "PDF_datasets/" + folder_name
    #Check if the requested folder name exists
    if not os.path.isdir(pdf_path):
        print("This folder doesn't exist. Choose an existing folder.")
        #raise ValueError("This folder doesn't exist. Choose an existing folder.")
        return
    
    X_train = pd.read_hdf(pdf_path + "/X_train.h5", key='df')
    y_train = pd.read_hdf(pdf_path + "/y_train.h5", key='df')
    X_val   = pd.read_hdf(pdf_path + "/X_val.h5", key='df')
    y_val   = pd.read_hdf(pdf_path + "/y_val.h5", key='df')
    X_test  = pd.read_hdf(pdf_path + "/X_test.h5", key='df')
    y_test  = pd.read_hdf(pdf_path + "/y_test.h5", key='df')
    return X_train, y_train, X_val, y_val, X_test, y_test

def test_model(X_test, y_test, model): #Test the model on the test dataset, prediction on the test set
    start_time = time.time()
    xgb_test = xgb.DMatrix(X_test, y_test)
    y_pred_proba_test = model.predict(xgb_test)
    y_pred_test = [np.argmax(line) for line in y_pred_proba_test]
    
    y_pred_proba_test_list = y_pred_proba_test.tolist()
    
    #print("y_pred_test:")
    #print(y_pred_test)

    #Finding the accuracy of the top3 guesses:
    y_pred_proba_top3 = [] #[the largest 3 %s, ...]
    for guesslist in y_pred_proba_test: #Finds the five largest percentages
        guesslist.sort()
        top3 = guesslist[-3::]
        y_pred_proba_top3.append(top3)
    
    y_pred_proba_top3_index = [] #[[index for the largest 3 %], ... ]
    for top3list in range(len(y_pred_proba_top3)): #Finds the indexes of the five largest percentages, meaning which 5 guesses
        top3index = []
        for percent in y_pred_proba_top3[top3list]:
            indexofguess = y_pred_proba_test_list[top3list].index(percent)
            top3index.append(indexofguess)
        y_pred_proba_top3_index.append(top3index)
        
    correct_guesses_num = 0
    for correctguessindex in range(len(y_test)): 
        for guess in y_pred_proba_top3_index[correctguessindex]: #Checks if any of the guesses are the right guess
            if y_test[correctguessindex] == guess:
                correct_guesses_num += 1
    top3accuracy = 100 * correct_guesses_num / len(y_test)
    #print(f"There were {correct_guesses_num} correct guesses in top 3 guesses out of {len(y_test)}")
    
    #Finding the accuracy of the top5 guesses:
    y_pred_proba_top5 = [] #[The largest 5 %s, ...]
    for guesslist in y_pred_proba_test: #Finds the five largest percentages
        guesslist.sort()
        top5 = guesslist[-5::]
        y_pred_proba_top5.append(top5)
    
    y_pred_proba_top5_index = [] #[[index for the largest 5 %s], ... ]
    for top5list in range(len(y_pred_proba_top5)): #Finds the indexes of the five largest percentages, meaning which 5 guesses
        top5index = []
        for percent in y_pred_proba_top5[top5list]:
            indexofguess = y_pred_proba_test_list[top5list].index(percent)
            top5index.append(indexofguess)
        y_pred_proba_top5_index.append(top5index)
        
    correct_guesses_num = 0
    for correctguessindex in range(len(y_test)): 
        for guess in y_pred_proba_top5_index[correctguessindex]: #Checks if any of the guesses are the right guess
            if y_test[correctguessindex] == guess:
                correct_guesses_num += 1
    top5accuracy = 100 * correct_guesses_num / len(y_test)
    #print(f"There were {correct_guesses_num} correct guesses in top 5 guesses out of {len(y_test)}")
  
    print("Percent guessed structures in test set:", str(accuracy_score(y_test, y_pred_test)*100)[0:5], "%")
    print(f"Percent guessed structures in top3 in test set: {str(top3accuracy)[0:5]} %")
    print(f"Percent guessed structures in top5 in test set: {str(top5accuracy)[0:5]} %")
    print("Time spent on predicting with model:", str((time.time()-start_time)/60)[0:6], " min")
    return None

def load_exp_pdf(exp_filename):
    exp_path = "ExperimentalData/"
    if not os.path.isfile(exp_path + exp_filename):
        print("The file is not located in the /ExperimentalData folder")
        return None
    else:
        for skiprows in range(50):
            try:
                label = np.loadtxt(exp_path+exp_filename, skiprows = skiprows)
                break
            except:
                pass
        #if not label:
        #    print("File was not loaded")    
        
        xgrid, xyz_pdf = label.T[0], label.T[1]
        
        if xgrid[0] != 0.0:
            #If it does not start from 0
            start_of_xgrid = [n/100 for n in range(int(min(xgrid)/0.01))] #0.0, 0.01, 0.02, ..., 
            xgrid_new = np.append(start_of_xgrid, xgrid)                  #new x_grid
            xyz_pdf = np.append(np.zeros(int(min(xgrid)/0.01)), xyz_pdf)
            xgrid = xgrid_new
        
        if xgrid[1] - xgrid[0] == 0.01:
            xyz_pdf = xyz_pdf[::10]
            xgrid = xgrid[::10]
        
        xyz_pdf = xyz_pdf[0:300]
        xgrid = xgrid[0:300]
        
        if len(xyz_pdf) < 300:
            while len(xyz_pdf) < 300:
                xyz_pdf = np.append(xyz_pdf, np.array([0]))
                xgrid = np.append(xgrid, np.array([xgrid[-1]+0.1]))
        
        xyz_pdf_raw = xyz_pdf
        xyz_pdf = xyz_pdf / max(xyz_pdf)
        
        data = np.column_stack([xgrid, xyz_pdf])
        datafile_path = "ExperimentalData/Cleaned/" + exp_filename
        np.savetxt(datafile_path , data, fmt=['%2.1f','%2.5f'])
    
    return xgrid, xyz_pdf, xyz_pdf_raw

def plot_loaded_PDF(xgrid, xyz_pdf, xyz_pdf_raw, exp_filename):
    plt.figure(figsize = (10, 5))
    plt.plot(xgrid, xyz_pdf, label = "Imported experimental PDF")
    plt.plot(xgrid, xyz_pdf_raw, label = "Raw experimental PDF")
    plt.title(exp_filename)
    plt.xlabel("r [Å]")
    plt.ylabel("G(r) [a.u.]")
    plt.legend()
    plt.show()
    return None

def model_predict_cluster(xyz_path, xyz_pdf, exp_filename, model, sorted_filenames_flat, Qmin, Qmax, Qdamp):
    """
    Takes a PDF with 300 points, throws it into the model, prints the 5 best guesses.
    """
    xyz_pdf = xyz_pdf/max(xyz_pdf)
    xyz_pdf = np.append(xyz_pdf, Qmin)
    xyz_pdf = np.append(xyz_pdf, Qmax)
    xyz_pdf = np.append(xyz_pdf, Qdamp)

    mad = xyz_pdf.reshape((1,-1))
    xgb_test = xgb.DMatrix(mad) #, label = columnsliste)

    percentages = model.predict(xgb_test)
    percentages_list = percentages[0].tolist()
    indexn = percentages_list.index(max(percentages[0]))
    
    indexfilename = sorted_filenames_flat[indexn]

    percentages_list_sorted = sorted(percentages_list)
    top5list = percentages_list_sorted[::-1][0:5]

    print(f"The model was given cluster: {exp_filename}")
    print(f"The model predicts the following:")
    
    indexn_list = [] #These will be the indices of the top 5 guesses
    guess_filenames = []
    for n in range(len(top5list)):
        indexn = percentages_list.index(top5list[n])
        indexn_list.append(indexn)
        
        indexfilename = sorted_filenames_flat[indexn]
        guess_filenames.append(indexfilename)
        
        print(f"Prediction {n+1} with {str(top5list[n]*100)[0:4]} % is: {indexfilename}")
    
    #So we want to make a directory for the results
    os.makedirs("Results/" + "Results_" + exp_filename[0:-3], exist_ok=True)
    #Then, copy over the 5 guessed xyz files to the results folder
    shutil.copy(xyz_path+str(guess_filenames[0]), "Results/" + "Results_" + exp_filename[0:-3] + "/1_guess_" + str(guess_filenames[0]))
    shutil.copy(xyz_path+str(guess_filenames[1]), "Results/" + "Results_" + exp_filename[0:-3] + "/2_guess_" + str(guess_filenames[1]))
    shutil.copy(xyz_path+str(guess_filenames[2]), "Results/" + "Results_" + exp_filename[0:-3] + "/3_guess_" + str(guess_filenames[2]))
    shutil.copy(xyz_path+str(guess_filenames[3]), "Results/" + "Results_" + exp_filename[0:-3] + "/4_guess_" + str(guess_filenames[3]))
    shutil.copy(xyz_path+str(guess_filenames[4]), "Results/" + "Results_" + exp_filename[0:-3] + "/5_guess_" + str(guess_filenames[4]))
    
    print("The five predicted clusters have been saved to Results/" + str(exp_filename[0:-3]) + "/")

    return indexn_list, guess_filenames, top5list

def fit(Qmin, Qmax, Qdamp, cluster, PDFfile, plot):
    # Create a PDF contribution as before
    pdfprofile = Profile()
    pdfparser = PDFParser()
    pdfparser.parseFile(PDFfile)
    pdfprofile.loadParsedData(pdfparser)
    pdfprofile.setCalculationRange(xmin = 1.5, xmax = 20, dx=0.01)

    # Setup the PDFgenerator that calculates the PDF from the model
    #Generator for first cluster
    pdfgenerator = DebyePDFGenerator("G")
    pdfgenerator._calc.evaluatortype = 'OPTIMIZED'

    #Input the data files
    #cluster = loadStructure(stru)
    pdfgenerator.setStructure(cluster, periodic=False)

    # Add the profile and generator the the PDFcontribution
    pdfcontribution = FitContribution("pdf")
    pdfcontribution.setProfile(pdfprofile, xname="r")
    pdfcontribution.addProfileGenerator(pdfgenerator)
    pdfcontribution.setEquation("scale*G")

    # Moving on
    recipe = FitRecipe()
    recipe.addContribution(pdfcontribution)
    recipe.addVar(pdfcontribution.scale, 0.1, tag = "scale")
    recipe.restrain("scale", lb=0.1, ub=1e99, sig=0.001)
    pdfgenerator.qdamp.value = Qdamp
    pdfgenerator.setQmax(Qmax)
    pdfgenerator.setQmin(Qmin)

    # Add ADP for the cluster
    phase_molecule = pdfgenerator.phase
    atoms1 = phase_molecule.getScatterers()

    #Make latices to the two phases
    lat = phase_molecule.getLattice()
    #Make new variable zoomscale
    recipe.newVar("zoomscale", 1.00, tag = "lat")
    recipe.constrain(lat.a, 'zoomscale')
    recipe.constrain(lat.b, 'zoomscale')
    recipe.constrain(lat.c, 'zoomscale')

    # We create the variables of ADP and assign the initial value to them. In this
    # example, we use isotropic ADP for all atoms
    Biso = recipe.newVar("Biso", value=0.2, tag = 'ADP')
    recipe.restrain(Biso, lb=0, ub=2, sig = 0.01)
 
    # For all atoms in the structure model, we constrain their Biso according to their species 
    for atom in atoms1:
        if atom.element == cluster.element[0]:
            recipe.constrain(atom.Biso, Biso)
        
    recipe.clearFitHooks()
    #     Tune PDF
    recipe.fix("all")
    recipe.free("lat")
    leastsq(recipe.residual, recipe.getValues())   
    recipe.free("scale")
    leastsq(recipe.residual, recipe.getValues())   
    recipe.free("ADP")
    leastsq(recipe.residual, recipe.getValues())   
    
    res = FitResults(recipe)
    rfactor = res.rw
    
    # All this should be pretty familiar by now.
    r = recipe.pdf.profile.x
    g = recipe.pdf.profile.y
    gcalc = recipe.pdf.profile.ycalc
    diffzero = -0.8 * max(g) * np.ones_like(g)
    diff = g - gcalc + diffzero
    
    if plot:
        output_notebook()
        tools = "hover, box_zoom, undo, crosshair"
        p = figure(tools=tools, background_fill_color="darkgray")

        p.scatter(r,g,color='blue',legend_label="G(r) Data")
        p.line(r, gcalc,color='red',legend_label="G(r) Fit")
        p.line(r, diff,color='green',legend_label="G(r) diff")
        p.line(r, diffzero,color='black')
        show(p)
    
        res.printResults()
    return rfactor, r, g, gcalc, diff

def fit_top3(guess_filenames, xyz_path, exp_filename, Qmin, Qmax, Qdamp):
    print("\nFit of the first prediction: " + str(guess_filenames[0]))
    Bi1 = loadStructure(xyz_path + guess_filenames[0])
    PDFFile = "ExperimentalData/Cleaned/" + exp_filename
    rfactor, r, g, gcalc, diff = fit(Qmin, Qmax, Qdamp = 0.03, cluster = Bi1, PDFfile = PDFFile, plot = True)

    print("\nFit of the second prediction: " + str(guess_filenames[1]))
    Bi2 = loadStructure(xyz_path + guess_filenames[1])
    PDFFile = "ExperimentalData/Cleaned/" + exp_filename
    rfactor, r, g, gcalc, diff = fit(Qmin, Qmax, Qdamp = 0.03, cluster = Bi2, PDFfile = PDFFile, plot = True)

    print("\nFit of the third prediction: " + str(guess_filenames[2]))
    Bi3 = loadStructure(xyz_path + guess_filenames[2])
    PDFFile = "ExperimentalData/Cleaned/" + exp_filename
    rfactor, r, g, gcalc, diff = fit(Qmin, Qmax, Qdamp = 0.03, cluster = Bi3, PDFfile = PDFFile, plot = True)
    return None
