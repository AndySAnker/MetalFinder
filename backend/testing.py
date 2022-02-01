### Backend for testing

###Import packages 
import sys, os, os.path, h5py, time, shutil

from os import walk
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize.minpack import leastsq
from diffpy.Structure import loadStructure
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.fitbase import FitRecipe, FitResults

from shutil import copy

###list of functions here
#-Importer PDF dataset
#-Load XGBoost model
#-Test XGBoost model på test PDF set
#-Load enkelt experimentel PDF
#-Test XGBoost model på experimentel PDF

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

def test_model(X_test, y_test, model): #Test the model on the test dataset, prediction on the test set. Model has to be named model
    start_time = time.time()
    xgb_test = xgb.DMatrix(X_test, y_test)
    y_pred_proba_test = model.predict(xgb_test)
    y_pred_test = [np.argmax(line) for line in y_pred_proba_test]
    
    y_pred_proba_test_list = y_pred_proba_test.tolist()
    
    #Finding the accuracy of the top3 guesses:
    y_pred_proba_top3 = [] #[de største 3 procenter, de største 3 procenter, ...]
    for guesslist in y_pred_proba_test: #Finds the five largest percentages
        guesslist.sort()
        top3 = guesslist[-3::]
        y_pred_proba_top3.append(top3)
    
    y_pred_proba_top3_index = [] #[[index for de største 3 procenter], ... ]
    for top3list in range(len(y_pred_proba_top3)): #Finds the indexes of the five largest percentages, meaning which 5 guesses
        top3index = []
        for percent in y_pred_proba_top3[top3list]:
            indexofguess = y_pred_proba_test_list[top3list].index(percent)
            top3index.append(indexofguess)
        y_pred_proba_top3_index.append(top3index)
        
    correct_guesses_num = 0
    for correctguessindex in range(len(y_test)): #Nu skal vi se om det rigtige gæt kan findes i maskinens 5 gæt
        for guess in y_pred_proba_top3_index[correctguessindex]: #Checks if any of the guesses are the right guess
            if y_test[correctguessindex] == guess:
                correct_guesses_num += 1
    top3accuracy = 100 * correct_guesses_num / len(y_test)
    #print(f"There were {correct_guesses_num} correct guesses in top 3 guesses out of {len(y_test)}")
    
    #Finding the accuracy of the top5 guesses:
    y_pred_proba_top5 = [] #[de største 5 procenter, de største 5 procenter, ...]
    for guesslist in y_pred_proba_test: #Finds the five largest percentages
        guesslist.sort()
        top5 = guesslist[-5::]
        y_pred_proba_top5.append(top5)
    
    y_pred_proba_top5_index = [] #[[index for de største 5 procenter], ... ]
    for top5list in range(len(y_pred_proba_top5)): #Finds the indexes of the five largest percentages, meaning which 5 guesses
        top5index = []
        for percent in y_pred_proba_top5[top5list]:
            indexofguess = y_pred_proba_test_list[top5list].index(percent)
            top5index.append(indexofguess)
        y_pred_proba_top5_index.append(top5index)
        
    correct_guesses_num = 0
    for correctguessindex in range(len(y_test)): #Nu skal vi se om det rigtige gæt kan findes i maskinens 5 gæt
        for guess in y_pred_proba_top5_index[correctguessindex]: #Checks if any of the guesses are the right guess
            if y_test[correctguessindex] == guess:
                correct_guesses_num += 1
    top5accuracy = 100 * correct_guesses_num / len(y_test)
    #print(f"There were {correct_guesses_num} correct guesses in top 5 guesses out of {len(y_test)}")
  
    print("Percent guessed structures in test set:", str(accuracy_score(y_test, y_pred_test)*100)[0:5], "%")
    print(f"Percent guessed structures in top3 in test set: {str(top3accuracy)[0:5]} %")
    print(f"Percent guessed structures in top5 in test set: {str(top5accuracy)[0:5]} %")
    print("Time spent on predicting with model:", str((time.time()-start_time)/60)[0:6], " min")
    return None #accuracy_score(y_test, y_pred_test)*100

def load_exp_pdf(exp_filename): #load_file_3
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
        #    print("Filen blev ikke loadet")    
        
        xgrid, xyz_pdf = label.T[0], label.T[1]
        
        if xgrid[0] != 0.0:
            #Oh shit, den starter ikke fra nul
            start_of_xgrid = [n/100 for n in range(int(min(xgrid)/0.01))] #Laver en liste der tæller op fra 0.0, 0.01, 0.02, ..., 
            xgrid_new = np.append(start_of_xgrid, xgrid)                  #Den nye xgrid som er som den skal være
            xyz_pdf = np.append(np.zeros(int(min(xgrid)/0.01)), xyz_pdf)  #Den nye PDF med 0'er sat ind til at starte med
            xgrid = xgrid_new
        
        if xgrid[1] - xgrid[0] == 0.01: #Hvis ikke den er nyquist sampled
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
        
        #Hver gang et lys slukkes, så tændes der et nyt
        #Og hver gang en eksperimental datafil loades, så gemmes den i børneformat, så DiffPy kan følge med
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

def model_predict_cluster(xyz_path, xyz_pdf, exp_filename, model, sorted_filenames_flat, Qmin, Qmax, Qdamp): #NEW 06/05/2021
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
        
        print(f"Guess {n+1} with {str(top5list[n]*100)[0:4]} % is: {indexfilename}")
    
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

def fitPDF(guess_filename, exp_filename, xyz_path, Qmin, Qmax, Qdamp):
    
    #print(guess_filenames[0]) #Det her er filnavnet af et godt gæt
    #exp_pdf_curated          #Det her er PDFen som vi skal fitte til
    
    diffpy_exp_path = "ExperimentalData/Cleaned/"
    
    structureFile = xyz_path + guess_filename
    dataFile = diffpy_exp_path + exp_filename   #start loading the ones that we're going to save with load_data_2 automatically
    
    r_min, r_max, r_step = 0, 30, 0.1 #inkluder i funktionsparametrene? Nej det tror jeg ikke
    #Qmin, Qmax, Qdamp = 0.8, 27, 0.04

    # Define the PDF contribution which will hold the data and structure
    pdfcontribution = PDFContribution("pdf")

    # Give the data to the contribution and set the r-range over which we'll fit
    pdfcontribution.loadData(dataFile) #Her loader diffpy selv PDFen - det kan den ikke finde ud af
    pdfcontribution.setCalculationRange(xmin=r_min, xmax=r_max, dx=r_step)
    
    # Add the structure from our xyz file to the contribution. Since the structure
    # model is non-periodic, we need to specify the periodic=False here to get the right PDF
    structure = loadStructure(structureFile) #Loader xyz-filen som er modellens gæt
    pdfcontribution.addStructure("Cluster", structure, periodic=False)
    
    # Set Qmin and Qmax for the data
    pdfcontribution.Cluster.setQmin(Qmin)
    pdfcontribution.Cluster.setQmax(Qmax)
    
    # The FitRecipe does the work of calculating the PDF with the fit variable that we give it.
    recipe = FitRecipe()
    
    # give the PDFContribution to the FitRecipe
    recipe.addContribution(pdfcontribution)
    
    # Qdamp is fixed based on prior information about our beamline.
    recipe.addVar(pdfcontribution.qdamp, Qdamp, fixed=True)
    
    # We add variables for the overall scale of the PDF and a delta2
    # parameter for correlated motion of neighboring atoms.
    recipe.addVar(pdfcontribution.scale, 1)
    
    # We create the variables of ADP and assign the initial value to them. In this
    # example, we use isotropic ADP for all atoms
    MBiso = recipe.newVar("M_Biso", value=0.3, tag='M_iso')
    
    # For all atoms in the structure model, we constrain their Biso according to 
    # their species  
    atoms = pdfcontribution.Cluster.phase.getScatterers()
    for atom in atoms:
        recipe.constrain(atom.Biso, MBiso)
        recipe.restrain(atom.Biso, lb=0, ub=3, sig=0.001)
        
    # We create a zoomscale factor which stretches the structure model, this is 
    # useful when you want to fit the bond length. Note that the relative position
    # of atoms are not changed during the refinements
    Lat = recipe.newVar('Lat', value=1, tag='lat')
    
    #We're trying without the "Lat" lines that don't work
    
    # Here is a simple way to assign the zoomscale to the structure. Note that this
    # only works for NON-PERIODIC structure 
    lattice = pdfcontribution.Cluster.phase.getLattice()
    recipe.constrain(lattice.a, Lat) 
    recipe.constrain(lattice.b, Lat)
    recipe.constrain(lattice.c, Lat)
    
    # We can now execute the fit using scipy's least square optimizer.
    recipe.clearFitHooks()
    recipe.fix('all')
    recipe.free("lat") #'scale', 
    leastsq(recipe.residual, recipe.getValues())
    
    recipe.free("scale") #'scale', 
    leastsq(recipe.residual, recipe.getValues())
    
    results = FitResults(recipe)
    recipe.free("M_iso")
    leastsq(recipe.residual, recipe.getValues())
    results = FitResults(recipe)

    #print("FIT RESULTS\n")
    #print(results)
    
    # Plot the observed and refined PDF.
    # Get the experimental data from the recipe
    r = recipe.pdf.profile.x
    gobs = recipe.pdf.profile.y
    
    # Get the calculated PDF and compute the difference between the calculated and
    # measured PDF
    gcalc = recipe.pdf.evaluate()
    gcalc = gcalc / max(gcalc)
    gobs = gobs / max(gobs)
    baseline = 1.1 * gobs.min() -0.6
    gdiff = gobs - gcalc #GCALC ER DEN FITTEDE PDF!!!
    
    values = recipe.getValues()
    fitparameternames = recipe.getNames()
    print("\n" + "Fit parameters for: " + str(guess_filename))

    print(str(fitparameternames[0]) + ": " + str(values[0]))
    print(str(fitparameternames[1]) + ": " + str(values[1]))
    print(str(fitparameternames[2]) + ": " + str(values[2]))

    #Somehow, recipe.pdf.evaluate() sometimes only gives 286 points. We're just putting zeros at the end
    if len(gcalc) < 300:
        gcalc = np.append(gcalc, np.zeros(300 - len(gcalc)))
    
    return gcalc, results.rw #Det er rw #Skal der nogle fit parametre med, måske?

def plot_PDF(xgrid_curated, exp_pdf_curated, filename, indexn_list, top5list, guess_filenames, gcalc1, gcalc2, gcalc3, rw1, rw2, rw3):
    
    plt.figure(figsize=(13, 9))
    
    offset = 0.9
    
    #Plotter de fittede PDFer
    plt.plot(xgrid_curated, gcalc1[0:300]+5*offset, label = f"Fitted PDF guess 1: Model certainty: {str(100*top5list[0])[0:5]} %, Rw: {str(rw1)[0:4]}", color = "red")
    plt.plot(xgrid_curated, gcalc2[0:300]+3*offset, label = f"Fitted PDF guess 2: Model certainty: {str(100*top5list[1])[0:5]} %, Rw: {str(rw2)[0:4]}", color = "red")
    plt.plot(xgrid_curated, gcalc3[0:300]+1*offset, label = f"Fitted PDF guess 3: Model certainty: {str(100*top5list[2])[0:5]} %, Rw: {str(rw3)[0:4]}", color = "red")
    
    #Plotter den eksperimentelle PDF
    plt.plot(xgrid_curated, exp_pdf_curated+5*offset, color = "C0") #Experimental
    plt.plot(xgrid_curated, exp_pdf_curated+3*offset, color = "C0") #Experimental
    plt.plot(xgrid_curated, exp_pdf_curated+1*offset, label = "Normalised PDF from experimental data", color = "C0") #Experimental
    
    #Regner diff mellem fittet PDF og eksperimentel PDF ud
    gdiff1 = exp_pdf_curated - gcalc1[0:300]
    gdiff2 = exp_pdf_curated - gcalc2[0:300]
    gdiff3 = exp_pdf_curated - gcalc3[0:300]
    
    #Plotter gdiff
    diff_offset = 0.25
    plt.plot(xgrid_curated, gdiff1+4*offset+diff_offset, color = "green", label = "Difference between fitted PDF and experimental PDF") #"green"
    plt.plot(xgrid_curated, gdiff2+2*offset+diff_offset, color = "green") #"orange"
    plt.plot(xgrid_curated, gdiff3+0*offset+diff_offset, color = "green") #"red"
    
    #Plotter vertikale linjer ved gdiff
    plt.hlines(4*offset+diff_offset, xmin=0, xmax=30, colors="grey", linestyles='dotted')
    plt.hlines(2*offset+diff_offset, xmin=0, xmax=30, colors="grey", linestyles='dotted')
    plt.hlines(0*offset+diff_offset, xmin=0, xmax=30, colors="grey", linestyles='dotted')
    
    #plt.title(f"PDF from experimental data: \"{filename}\"")
    plt.title(f"Tree-based model top3 predictions on experimental pdf: \"{filename}\"")
    
    if len(guess_filenames[0]) < 50:
        plt.text(x = 18, y = 4.35, s = guess_filenames[0])
    else:
        plt.text(x = 18, y = 4.35, s = guess_filenames[0][0:50])
    
    if len(guess_filenames[1]) < 50:
        plt.text(x = 18, y = 2.55, s = guess_filenames[1])
    else:
        plt.text(x = 18, y = 2.55, s = guess_filenames[1][0:50])
        
    if len(guess_filenames[2]) < 50:
        plt.text(x = 18, y = 0.75, s = guess_filenames[2])
    else:
        plt.text(x = 18, y = 0.75, s = guess_filenames[2][0:50])
        
    #plt.text(x = 18, y = 2.55, s = guess_filenames[1])
    #plt.text(x = 18, y = 0.75, s = guess_filenames[2])

    plt.xlabel("r [Å]")
    plt.ylabel("G (r)")
    plt.legend()
    plt.savefig("Results/" + "Results_" + str(filename)[0:-3] + "/Results_" +str(filename)[0:-3] + ".png", bbox_inches='tight', dpi = 400)
    plt.show()