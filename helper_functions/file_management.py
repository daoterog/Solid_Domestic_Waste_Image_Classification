import os
import shutil
import errno
import zipfile

import random

import pandas as pd
import numpy as np

# from sklearn.externals import joblib
import joblib

def copy_folder(path, destination):

    """
    Copies the experiment directory into the specifified destination. It allows
    the user to fill a description in order to give further insight of the 
    experimentation.

    Args:
        path: path to the experiment folder.
        destination: path to the destination folder.
    """

    comment = str(input('Enter the experiment description:'))

    # Extract experiment name
    experiment = os.path.basename(path)

    np.savetxt(os.path.join(path,experiment + '.txt'),[comment], fmt = '%10s')

    copytree(experiment,path,destination)

def bring_data(path_list, root='data'):
    """
    Brings zip folders and unzips them into the specified root. 

    Output:
        path_list: list of paths to each of the compressed folders stored
            in the drive.
        root: path in which the compressed folder will be unzipped.
    """

    # Delete directories to avoid over-riding files
    try:
        shutil.rmtree(root)
    except Exception:
        print('Not found')

    # Create root directory
    os.mkdir(root)
    
    # Copying zip files
    for filepath in path_list:
        
        # Extract Category Names
        category = os.path.basename(filepath)

        # Copy zip files
        shutil.copy(filepath,root)

        # Unzip the files
        local_zip_folder = os.path.join(root,category)
        with zipfile.ZipFile(local_zip_folder, 'r') as zip_ref:
            zip_ref.extractall(root)

        # Remove zip folder
        os.remove(local_zip_folder)

def split_images(root='data/', train_size=0.7, test_proportion=0.5):
    """
    Splits the images into train and test sets. This allows us to perform
    certain analysis after evaluating the model.

    Args:
        root: The root path in which is stored the data
        train_size: proportion of the data that will correspond to the training
            set.
        test_proportion: proportion of the remaining data (after splitting the 
            training portion) that will correspond to the test set (eg. if 
            train_size=0.7 then the test_size will be of 0.15 because 
            (1-0.7)*0.5=0.15)
    """

    train_path = os.path.join(root,'train')
    validation_path = os.path.join(root,'validation')
    test_path = os.path.join(root,'test')

    # Delete directories to avoid over-riding shuffle
    try:
        shutil.rmtree(train_path)
        shutil.rmtree(validation_path)
        shutil.rmtree(test_path)
    except Exception:
        print('Not found')

    # Create train and test directories
    os.mkdir(train_path)
    os.mkdir(validation_path)
    os.mkdir(test_path)
    
    for category in os.listdir(root):
        if category == 'test' or category == 'train' or category == 'validation':
            pass
        else:
            # Extract the image paths
            category_path = os.path.join(root,category)
            category_images = os.listdir(category_path)

            # Randomly choose train images
            category_train_images = random.choices(category_images, 
                                                    k=int(train_size * \
                                                          len(category_images))) 

            # Convert lists to set in order to get the remaining image paths
            category_set = set(category_images)
            train_set = set(category_train_images)
            category_validation_test_images = list(category_set - train_set)

            # Randomly choose validation images
            category_validation_images = random.choices(category_validation_test_images,
                                                         k=int((1-test_proportion) *
                                                               len(category_validation_test_images)))
            
            # Convert lists to sets in order to get the remaining image paths
            category_validation_test_images_set = set(category_validation_test_images)
            validation_set = set(category_validation_images)
            category_test_images = list(category_validation_test_images_set -
                                        validation_set)

            # Create category directory
            category_train_path = os.path.join(train_path,category)
            category_validation_path = os.path.join(validation_path,category)
            category_test_path = os.path.join(test_path,category)
            try:
                os.mkdir(category_train_path)
                os.mkdir(category_validation_path)
                os.mkdir(category_test_path)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    print('Directory already exist')
                else:
                    raise
            
            # Copy train files
            for image in category_train_images:
                src_path = os.path.join(category_path, image)

                if os.path.isfile(src_path):
                    dst_path = os.path.join(category_train_path,image)
                    shutil.copy(src_path, dst_path)

            # Copy validation files
            for image in category_validation_images:
                src_path = os.path.join(category_path, image)

                if os.path.isfile(src_path):
                    dst_path = os.path.join(category_validation_path,image)
                    shutil.copy(src_path, dst_path)
            
            # Copy test files
            for image in category_test_images:
                src_path = os.path.join(category_path, image)

                if os.path.isfile(src_path):
                    dst_path = os.path.join(category_test_path,image)
                    shutil.copy(src_path, dst_path)

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
        dir_path (str): target directory
    
    Returns:
        A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def storemodel(classifier, param_dict, dataset_key, X, y, path, model_name):

    """
    Stores a trained model (.joblib format) in the specified path.

    Args:
        classifier: used model to perform predictions.
        param_dict: models hyperparameter dictionary.
        dataset_key: dataset key.
        X: independent variables.
        y: dependent variable.
        path: path in which tje model is going to be stored.
        model_name: name of the model
    """ 

    # Set Parameters
    clf = classifier.set_params(**param_dict[dataset_key])

    # Train Model
    clf_fitted = clf.fit(X,y)

    # Save model
    joblib.dump(clf_fitted,path+'/'+model_name+'.joblib')


def storeresults(classifier, results, model_name, param_dict, 
                 param_title_dictionary, X, y, path):
    
    """ Creates a dataframe with tabulated results

    Args:
        classifier: used model to perform predictions.
        results: array of metric dictionaries.
        model_name: self explanatory.
        param_dict: models parameter grid.
        param_title_dictionary: model hyperparameter title (String).
        X: independent variables.
        y: dependent variables.
        path: path to load the model.

    Output: 
        df: Dataframe with results.
    """


    # Creating Dataframe that contains results
    df = pd.DataFrame({'Model':str,'Dataset':str,
                            'Mean_Precision':float,
                            'STD_Precision':float,
                            'Mean_Recall':float,
                            'STD_Recall':float,
                            'Mean_F1':float,
                            'STD_F1':float,
                            'Mean_AUC':float,
                            'STD_AUC':float},index = [0])

    for dataset_key in param_dict.keys():

        # Extract list
        precision = results[0][dataset_key]
        recall = results[1][dataset_key]
        f1 = results[2][dataset_key]
        auc = results[3][dataset_key]

        # Extract Statistic
        precision_mean = np.mean(precision)
        precision_std = np.std(precision)
        recall_mean = np.mean(recall)
        recall_std = np.std(recall)
        f1_mean = np.mean(f1)
        f1_std = np.std(f1)
        auc_mean = np.mean(auc)
        auc_std = np.std(auc)

        # Experiment model and hyperparameter
        name = model_name + '_' + param_title_dictionary[dataset_key]

        storemodel(classifier, param_dict, dataset_key, X, y, path, name)
        
        df = df.append({'Model':name,'Dataset':dataset_key,
                            'Mean_Precision':precision_mean,
                            'STD_Precision':precision_std,
                            'Mean_Recall':recall_mean,
                            'STD_Recall':recall_std,
                            'Mean_F1':f1_mean,
                            'STD_F1':f1_std,
                            'Mean_AUC':auc_mean,
                            'STD_AUC':auc_std},ignore_index=True)
        
    df.drop(index = 0, axis = 0, inplace = True)
        
    return df

def copytree(dir, src, dst, symlinks=False, ignore=None):
    """ 
    Used to copy a whole directory into a destination.

    Args:
        dir: name of the directory.
        src: source path.
        dst: destination path. 

    """

    dst = os.path.join(dst,dir)

    try:
        os.mkdir(dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exist')
        else:
            raise

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
