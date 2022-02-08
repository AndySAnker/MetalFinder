### Backend for training

###Import packages 
import sys, os, os.path, h5py, time

from os import walk
import pandas as pd
import xgboost as xgb

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

def get_training_data_backend(xyz_folder_name):
	#Sorted filenames:
	xyz_path = "xyz_files/" + xyz_folder_name + "/" #Path where the xyz files are
	sorted_filenames = sort_filenames(xyz_path, print_level = False)
	sorted_filenames_flat = [item for sublist in sorted_filenames for item in sublist]
	
	#Instantiate the PDF_generator class
	#PDF_obj = PDF_generator()

	return sorted_filenames_flat, xyz_path #Eller bare lav den her global? Det giver et simplere dokument måske idk

xyz_folder_name = "natoms200" #Change if using other xyz-file folder
sorted_filenames_flat, xyz_path = get_training_data_backend(xyz_folder_name)

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

def ML(X_train, y_train, X_val, y_val, model, n_threads, n_epochs, xyz_path): #Used for training a model
    
    #xyz_path is defined in the start of the notebook. According to the xyz-files folder chosen
    sorted_filenames = sort_filenames(xyz_path, print_level = False)
    sorted_filenames_flat = [item for sublist in sorted_filenames for item in sublist]

    # Setting up parameters
    xgb_params = {}
    xgb_params['learning_rate'] = 0.15 #Learning rate
    xgb_params['objective'] = 'multi:softprob'       #Multi-class target feature
    xgb_params['num_class'] = len(sorted_filenames_flat) #structures #no.of unique values in the target class not inclusive of the end value
    xgb_params['n_jobs'] = n_threads
    xgb_params['eval_metric'] = ['mlogloss']
    epochs = n_epochs
    
    #xgb_params['tree_method'] = "gpu_hist" #If these are enabled, the GPU is used - for training
    #xgb_params['predictor'] = 'gpu_predictor' #If these are enabled, the GPU is used - for guessing
    xgb_params['max_depth'] = 3 
    #xgb_params["single_precision_histogram"] = True #debug
    #xgb_params["subsample"] = 0.1 #debug
    #xgb_params["sampling_method"] = "gradient_based" #debug
    #training the model
    start_time = time.time()
    
    xgb_train = xgb.DMatrix(X_train, y_train) #Turns Pandas dataframe into an xgboost format
    xgb_val = xgb.DMatrix(X_val, y_val)
    store = {}
    evallist = [(xgb_train,'train'),(xgb_val, 'val')]
    print ("Time spent on making data ready:", (time.time()-start_time)/60, " min")

    #Train the model:
    start_time = time.time()
    print ("Training model")
    
    model = None #Disable this if continuing training of a model
    model = xgb.train(xgb_params, xgb_train, epochs, evallist, evals_result=store, verbose_eval=1, early_stopping_rounds=5, xgb_model=model)
    print ("Time spent on training model:", (time.time()-start_time)/60, " min")
    return model#, (time.time()-start_time)/60
