import os
import shutil

import pandas as pd
import numpy as np
from decimal import Decimal
import itertools

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import (auc, precision_score, recall_score, f1_score, 
                            average_precision_score, plot_precision_recall_curve,
                            roc_curve, classification_report, confusion_matrix)

import tensorflow as tf

def split_dataset(X_dict, y, train_size=0.6, val_test_split=0.5):

    """
    Splits dataset in train, validation, and test set.

    Args:
        X: independent variables.
        y: dependent variables.
        train_size: train size proportion.
        val_test_split: test proportion of the remaining portion of the dataset.

    Output:
        (X_train, y_train): train dataset tuple.
        (X_val, y_val): validation dataset tuple.
        (X_test, y_val): validation dataset tuple.
    """

    # Dictionary to store values
    data_train_dict = {}
    data_validation_dict = {}
    data_test_dict = {}

    # Loop through datasets
    for dataset_key in X_dict.keys():

        # Extract dataset
        X = X_dict[dataset_key]

        # Split datasets
        X_train, X_split, y_train, y_split = train_test_split(X, y, 
                                                            train_size=train_size)
        X_val, X_test, y_val, y_test = train_test_split(X_split, y_split, 
                                                        test_size=val_test_split)

        # Convert to tensor
        X_train = tf.reshape(tf.convert_to_tensor(X_train.to_numpy()), X_train.shape)
        y_train = tf.reshape(tf.convert_to_tensor(y_train.to_numpy()), y_train.shape)
        X_val = tf.reshape(tf.convert_to_tensor(X_val.to_numpy()), X_val.shape)
        y_val = tf.reshape(tf.convert_to_tensor(y_val.to_numpy()), y_val.shape)
        X_test = tf.reshape(tf.convert_to_tensor(X_test.to_numpy()), X_test.shape)
        y_test = tf.reshape(tf.convert_to_tensor(y_test.to_numpy()), y_test.shape)

        # Store datasets
        data_train_dict[dataset_key] = (X_train, y_train)
        data_validation_dict[dataset_key] = (X_val, y_val)
        data_test_dict[dataset_key] = (X_test, y_test)
        
    return data_train_dict, data_validation_dict, data_test_dict

def compare_historys(original_history, new_history, metric, initial_epochs, model_name, 
                     path):
    """
    Compares two TensorFlow model History objects.
    
    Args:
        original_history: History object from original model (before new_history)
        new_history: History object from continued model training (after 
            original_history).
        metric: metric with which the model was evaluated.
        initial_epochs: Number of epochs in original_history (new_history plot 
            starts from here).
        model_name: model name.
        path: path to store plot
    """
    
    # Get original history measurements
    acc = original_history.history[metric]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_" + metric]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history[metric]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_" + metric]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=100)
    fig.suptitle(model_name)

    # Acuracy learning curve
    ax[0].plot(total_acc, label='Training ' + metric)
    ax[0].plot(total_val_acc, label='Validation' + metric)
    ax[0].plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    ax[0].legend(loc='lower right')
    ax[0].set_title('Training and Validation ' + metric)
    ax[0].set_xlabel('epoch')

    # Loss learning curve
    ax[1].plot(total_loss, label='Training Loss')
    ax[1].plot(total_val_loss, label='Validation Loss')
    ax[1].plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    ax[1].legend(loc='upper right')
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('epoch')

    # Save Figure
    filename = 'learning_curve_' + model_name + '.jpg'
    fig.savefig(os.path.join(path, filename), dpi=100)
    plt.show()

def plot_loss_curves(history, model_name, path):
    """
    Returns separate loss curves for training and validation metrics.
    Args:
        history: TensorFlow model History object.
        model_name: model name.
        path: path to store plot
    """ 

    # Extract Values
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history[metric]
    val_accuracy = history.history['val_' + metric]

    epochs = range(len(history.history['loss']))

    # Creat Plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=100)
    fig.suptitle(model_name)

    # Plot loss
    ax[0].plot(epochs, loss, label='training_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(epochs, accuracy, label='training_' + metric)
    ax[1].plot(epochs, val_accuracy, label='val_' + metric)
    ax[1].set_title(metric)
    ax[1].set_xlabel('Epochs')
    ax[1].legend();

    # Save Figure
    filename = 'learning_curve_' + model_name + '.jpg'
    fig.savefig(os.path.join(path, filename), dpi=100)
    plt.show

def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.io.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img/255.
    else:
        return img

def print_most_wrong_prediction(wrong_preds, n):

    """
    Prints a sample of the most wrong predictions made.

    Args:
        wrong_preds: DataFrame with wrong predictions.
        n: number of images to print.
    """

    worng_preds_sample = wrong_preds.iloc[:n,:]

    for row in wrong_preds.itertuples():
        _, img_path, _, _, pred_prob, true_cn, pred_cn, _ = row

        img = load_and_prep_image(img_path, scale=True)
        plt.imshow(img)
        plt.title(f"{img_path}\nactual: {true_cn}, pred: {pred_cn} \nprob: {pred_prob:.2f}")
        plt.axis(False)
        plt.show()

def find_most_wrong_prediction_NN(test_data, y_true, y_pred, pred_probs, 
                                  class_names):
    
    """
    Creates a DataFrame compiling the images that were wrongfully classified by 
    the model.

    Args:
        test_data: Image Data Generator test images object
        y_true: true image labels.
        y_pred: predicted image labels.
        pred_probs: predicted probabilities.
        class_names: image labels class names.

    Output:
        wrong_preds: DataFrame with wrong prediction.
    """

    # Get the filenames of our test data
    filepaths = []
    for filepath in test_data.list_files('/content/data/test/*/*.jpg',
                                        shuffle=False):
        filepaths.append(filepath.numpy())

    # Create DataFrame
    pred_df = pd.DataFrame({
        'img_path': filepaths,
        'y_true': y_true,
        'y_pred': y_pred,
        'pred_prob': pred_probs.max(axis=1),
        'y_true_classname': [class_names[y] for y in y_true],
        'y_pred_classname': [class_names[y] for y in y_pred]
    })

    # Add column that indicates wether the prediction was right
    pred_df['pred_correct'] = pred_df.y_true == pred_df.y_pred

    # Get wrong predictions and sort them by their probability
    wrong_preds = pred_df[~pred_df.pred_correct].sort_values(by='pred_prob',
                                                                ascending=False)
    
    return wrong_preds

def evaluate_NN(model, test_data, model_name, path):

    """
    Evaluates Neural Networks performance by calculating most wrong predictions, 
    compiling accuracy, precision, recall, and f1-score, plotting confusion
    matrix, and previously mentioned scores barplots.

    Args:
        model: Neural Network.
        test_data: Image Data Generator test images object.
        model_name: name of the model
        path: path to compile experiments.

    Output:
        df_results: DataFrame with the results of each model.
        wrong_preds: DataFrame with wrong predictions.
    """

    # Make prediction with the model
    pred_probs = model.predict(test_data, verbose=1)

    # Get image labels
    y_labels = []
    for _, label in test_data.unbatch():
        y_labels.append(label.numpy().argmax())

    # Get predicted labels
    y_pred = pred_probs.argmax(axis = 1)

    # Get class names
    class_names = test_data.class_names

    # Find most wrong predictions
    wrong_preds = find_most_wrong_prediction_NN(test_data, y_labels, y_pred, 
                                                pred_probs, class_names)

    # Start plots
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

    # Find Across Class Results
    class_scores, accuracy, df_results = across_class_results(y_labels, 
                                                    y_pred, class_names,
                                                    model_name, fig, ax[1])

    # Confusion matrix
    make_confusion_matrix(y_labels, y_pred, fig, ax[0], accuracy,
                               class_names, norm=True)
    
    fig.suptitle(model_name)

    # Save Figure
    filename = 'cv_' + str(model_name) + '.jpg'
    fig.savefig(os.path.join(path,filename), dpi = 100)
    plt.show()

    # Make Second Plot
    precision_recall_barplot(class_scores, model_name, path)

    # Store Results
    df_results.to_csv(os.path.join(path,os.path.basename(path) + '.csv'))

    return df_results, wrong_preds

def create_wrong_prediction_image_dir(image_paths, dir):
  
  """
  Creates a provisional directory to store images wrongfully classified.
  
  Args:
    image_paths: list of image paths.
    dir: directory in which will be stored.
  """

  try:
      os.makedirs(dir)
  except OSError as e:
      if e.errno == errno.EEXIST:
          print('Directory already exist')
      else:
          raise

  for path in image_paths:
      shutil.copy('/content/'+path, os.path.join(dir,os.path.basename(path)))

def print_wrong_predictions(wrong_preds, n_images, image_generator, params, 
                            dir=None):
  
  """
  Prints sample of wrongfully predicted images.
  
  Args:
    wrong_preds: DataFrame of wrong classifications.
    n_images: number of images to be printed.
    image_generator: Image Generator Object.
    params: parameters used in the Image Generator.
    dir: provisional directory to store images.
  """

  # Extract Sample Images Paths
  sample_wrong_preds = wrong_preds.iloc[:n_images,]
  image_paths = sample_wrong_preds.img_path.tolist()

  # Create directory to store images

  # Make target directory
  if dir == None:
      dir = '/content/wrong_prediction_images/images'

  create_wrong_prediction_image_dir(image_paths, dir)

  # Load and process images
  data_gen = image_generator.flow_from_directory(directory=os.path.dirname(dir),
                                                 batch_size = n_images,
                                                 **params)

  # Extract images
  wrong_pred_images, _ = data_gen.next()

  # Plot images
  for i, row in enumerate(sample_wrong_preds.itertuples()):
      _, img_path, _, _, y_prob, y_true, y_pred, _ = row

      img = wrong_pred_images[i]

      plt.imshow(img[:,:,0])
      plt.title(f"{img_path}\nactual: {y_true}, pred: {y_pred} \nprob: {y_prob:.2f}")
      plt.axis(False)
      plt.show()

  # Delete provisional directory
  try:
      shutil.rmtree(os.path.dirname(dir))
  except Exception:
      print('Directory not found')

def across_class_results(y_true, y_pred, class_names, title, fig, ax):

    """
    Compiles the presicion, recall, and f1 scores of each class into a single 
    dataframe. It calculates the accuracy score as well.
    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        class_names: names of each target classes.
        title: title of the plot.
        fig: figure object for plotting.
        ax: axis object for plotting.
    Output:
        df_results: DataFrame compiling results for each class.
    """
    # Extract the classification report
    classification_report_dict = classification_report(y_true=y_true,
                                                    y_pred=y_pred,
                                                    output_dict=True)

    # Create empty dictionary
    class_f1_scores = {}
    class_precision = {}
    class_recall = {}

    # Create df columns
    df_columns = []
    df_values = []

    # Loop through the classification report items
    for key, value in classification_report_dict.items():
        if key == 'accuracy':
            accuracy = value
            df_columns = ['model_name',key] + df_columns
            df_values = [title, value] + df_values
            break
        else:

            # Extract Values
            name = class_names[int(Decimal(key))]
            f1_score = value['f1-score']; precision = value['precision']
            recall = value['recall']

            # Fill DataFrame Values
            df_columns.extend([name+'_f1_score', name+'_precision', name+'_recall'])
            df_values.extend([f1_score, precision, recall])

            # Construct score dictionaries for plotting
            class_f1_scores[name] = f1_score
            class_precision[name] = precision
            class_recall[name] = recall


    # Create df_results
    df_results = pd.DataFrame(df_values).transpose()
    df_results.columns = df_columns

    # Create DataFrame with dictionary
    class_scores = pd.DataFrame({"class_name": list(class_f1_scores.keys()),
                            "f1_score": list(class_f1_scores.values()),
                            'precision': list(class_precision.values()),
                            'recall': list(class_recall.values())})\
                            .sort_values("f1_score", ascending=False)

    # f1-score plot
    scores = ax.barh(range(len(class_scores)), class_scores["f1_score"].values)
    ax.set_yticks(range(len(class_scores)))
    ax.set_yticklabels(list(class_scores["class_name"]))
    ax.set_xlabel("f1-score")
    ax.set_title("F1-Scores for each Class")
    ax.invert_yaxis(); # reverse the order

    return class_scores, accuracy, df_results

def precision_recall_barplot(class_scores, title, path):

    """
    Plots precision and recall values for each class as barplots.

    Args:
        class_scores: DataFrame that contains necessary values.
        title: of the plot
        path: to load the data.
    """

    # Make Second Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), dpi=100)

    # precision plot
    scores = ax[0].barh(range(len(class_scores)), class_scores["precision"].values)
    ax[0].set_yticks(range(len(class_scores)))
    ax[0].set_yticklabels(list(class_scores["class_name"]))
    ax[0].set_xlabel("precision")
    ax[0].set_title("Precision Score for each Class")
    ax[0].invert_yaxis(); # reverse the order

    # recall plot
    scores = ax[1].barh(range(len(class_scores)), class_scores["recall"].values)
    ax[1].set_yticks(range(len(class_scores)))
    ax[1].set_yticklabels(list(class_scores["class_name"]))
    ax[1].set_xlabel("recall")
    ax[1].set_title("Recall Score for each Class")
    ax[1].invert_yaxis(); # reverse the order

    fig.suptitle(title)

    # Save Figure
    filename = 'pr_rc_' + str(title) + '.jpg'
    fig.savefig(os.path.join(path,filename), dpi = 100)
    plt.show()

def make_confusion_matrix(y_true, y_pred, fig, ax, accuracy, classes=None, 
                           norm=False, text_size=10): 

    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.
    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        fig: figure object for plotting.
        ax: axis object for plotting.
        accuracy: accuracy score calculated in across_class_results.
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).
        norm: normalize values or not (default=False).
        savefig: save confusion matrix to file (default=False).
    
    Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
    Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """  
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax, ax=ax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title=f"Confusion Matrix, Overall Accuracy = {accuracy:.3f}",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            ax.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        else:
            ax.text(j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
            
def find_most_wrong_prediction(image_filenames, test_indexes, y_true, y_pred, 
                               pred_prob, class_names, title, path, root='/data/'):
    
    """
    Creates a DataFrame compiling the images that were wrongfully classified by 
    the model.

    Args:
        image_filenames: filenames of the images used in the dataset.
        test_indexes: indexes used to test the classifier.
        y_true: true label of the image.
        y_pred: predicted label of the image.
        pred_prob: probability of the prediction.
        class_names: name of each predicted class.
        root: directory were the images are stored.

    Output:
        wrong_pred: DataFrame that compiles wrongfully predicted images.
    """

    # Extract values of interes
    test_filenames = image_filenames[test_indexes]
    img_paths = [os.path.join(root, filepath) for 
                                        filepath in test_filenames]

    # Create DataFrame
    pred_df = pd.DataFrame({
        'img_path': img_paths,
        'y_true': y_true,
        'y_pred': y_pred,
        'pred_prob': pred_prob,
        'y_true_classname': [class_names[int(y)] for y in y_true],
        'y_pred_classname': [class_names[int(y)] for y in y_pred]
    })
    
    # Add column that indicates wether the prediction was right
    pred_df['pred_correct'] = pred_df.y_true == pred_df.y_pred

    # Get wrong predictions and sort them by their probability
    wrong_preds = pred_df[~pred_df.pred_correct].sort_values(by='pred_prob',
                                                                ascending=False)
    
    # Save Dataframe as csv
    filename = 'wrong_pred_' + str(title) + '.jpg'
    wrong_preds.to_csv(os.path.join(path,filename + '.csv'))
    
    return wrong_preds
            
def multiclass_CV(classifier, k, X_dict, y, param_dict, param_title_dictionary, 
                  class_names, model_name, path, image_filenames):
    
    """
    Plots a confusion matrix and f1-scores calculated from the concatenation of 
    the results of the classifier over the whole dataset.
    Args:
        classifier: model.
        k: number of splits.
        X_dict: dataset dictionary.
        y: labels.
        param_dict: parameter dictionary for the model.
        param_title_dictionary: model hyperparameter title (String).
        model_name: self explanatory.
        path: to load the data.
    Output:
        df: DataFrame with the results of each model.
    """

    model_wrong_preds = {}

    df = pd.DataFrame()
    
    for dataset_key in X_dict.keys():

        X = X_dict[dataset_key].to_numpy()

        # Set parameters
        classifier.set_params(**param_dict[dataset_key])

        # Experiment Title
        title = model_name + '_' + dataset_key + '_' + \
                                            param_title_dictionary[dataset_key]

        skf = StratifiedKFold(n_splits=k)

        pred_labels = []
        true_labels = []
        test_indexes = []
        prob_labels = []

        for train_index, test_index in skf.split(X, y):

            # Split data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model and predict
            classifier.fit(X_train, y_train)
            pred_probs = classifier.predict_proba(X_test)
            y_pred = pred_probs.argmax(axis=1)
            y_prob = pred_probs.max(axis=1)

            # Make confusion matrix
            pred_labels.extend(y_pred)
            true_labels.extend(y_test)
            test_indexes.extend(test_index)
            prob_labels.extend(y_prob)

        # Find most wrong predictions
        model_wrong_preds[dataset_key] = find_most_wrong_prediction(image_filenames, 
                                            test_indexes, true_labels, pred_labels, 
                                            prob_labels, class_names, title, path)

        fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

        class_scores, accuracy, df_results = across_class_results(true_labels, 
                                                    pred_labels, class_names,
                                                    title, fig, ax[1])
        make_confusion_matrix(true_labels, pred_labels, fig, ax[0], accuracy,
                               class_names, norm=True)
        fig.suptitle(title)

        # Save Figure
        filename = 'cv_' + str(title) + '.jpg'
        fig.savefig(os.path.join(path,filename), dpi = 100)
        plt.show()

        # Make Second Plot
        precision_recall_barplot(class_scores, title, path)

        # Create and Store DataFrame
        df = pd.concat([df, df_results], ignore_index=True, axis=0)
        df.to_csv(os.path.join(path,os.path.basename(path) + '.csv'))

    return df, model_wrong_preds

def plotlearningcurve(model_name, param_dict, param_title_dictionary, score, 
                      train_sizes_dict, train_scores_mean_dict, 
                      train_scores_std_dict, test_scores_mean_dict,
                      test_scores_std_dict, path):
  
    """ 
    To plot the learning curve.

    Args:
        model_name: model name (String).
        param_dict: model parameter grid (dictionary).
        param_title_dictionary: model hyperparameter title (String).
        score: ylabel. Metric with which the model is evaluated (String).
        train_sizes_dict: train_sizes for each dataset considered (dictionary).
        train_scores_mean_dict: mean of the training scores for each train size 
            (dictionary).
        train_scores_std_dict: self explanatory.
        test_scores_mean_dict: self explanatory.
        test_scores_std_dict: self explanatory.
        path: path to load the graphs.

    """
    
    # Loop for each dataset
    for dataset_key in train_sizes_dict:

        train_sizes = train_sizes_dict[dataset_key]
        train_scores_mean = train_scores_mean_dict[dataset_key]
        train_scores_std = train_scores_std_dict[dataset_key]
        test_scores_mean = test_scores_mean_dict[dataset_key]
        test_scores_std = test_scores_std_dict[dataset_key]

        # Experiment Title
        title = model_name + '_' + dataset_key + '_' + \
                                            param_title_dictionary[dataset_key]

        plt.figure()
        plt.title(title)
        plt.xlabel('Training examples')
        plt.ylabel(score)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, 
                            color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
        plt.legend(loc="best")

        # Save Figure
        filename = 'learningcurve_'+ str(title) + '.jpg'
        plt.savefig(os.path.join(path,filename), dpi = 100)
        
        plt.show()

def learningcurve(classifier, X_dict, y, cv, param_dict, scoring, train_sizes):
    """ Calculate the learning curve values.

    Args:
        classifier: model used to perform the prediction
        X_dict: datasets dictionary
        y: labels
        cv: number of cross-validation splits
        param_dict: parameter dictionary
        scoring: metric used to evaluate cross validation
        train_sizes: specified train sizes

    Output:
        train_sizes_dict: train_sizes for each dataset considered (dictionary).
        train_scores_mean_dict: mean of the training scores for each train size 
            (dictionary).
        train_scores_std_dict: self explanatory.
        test_scores_mean_dict: self explanatory.
        test_scores_std_dict: self explanatory.
    """

    train_sizes_dict = {}
    train_scores_mean_dict = {}
    train_scores_std_dict = {}
    test_scores_mean_dict = {}
    test_scores_std_dict = {}

    for dataset_key in X_dict:

        X_train = X_dict[dataset_key]

        # Set Parameters
        classifier.set_params(**param_dict[dataset_key])

        train_sizes, train_scores, test_scores = learning_curve(classifier, 
                                    X_train, y, 
                                    cv = cv, 
                                    scoring = scoring, 
                                    train_sizes = train_sizes)

        train_scores_mean = np.mean(train_scores, axis = 1)
        train_scores_std = np.std(train_scores, axis = 1)
        test_scores_mean = np.mean(test_scores, axis = 1)
        test_scores_std = np.std(test_scores, axis = 1)

        train_sizes_dict[dataset_key] = train_sizes
        train_scores_mean_dict[dataset_key] = train_scores_mean
        train_scores_std_dict[dataset_key] = train_scores_std
        test_scores_mean_dict[dataset_key] = test_scores_mean
        test_scores_std_dict[dataset_key] = test_scores_std

    return (train_sizes_dict, train_scores_mean_dict, train_scores_std_dict,
            test_scores_mean_dict, test_scores_std_dict)

def plotrocauc(auc_list, fpr, tpr, tprs, mean_auc, mean_fpr,
               mean_tpr, title, path, fig, ax):
    """
    Plot the ROC curve.

    Args:
        auc_list: list of auc values.
        fpr: false positive rate list.
        tpr: true positive rate list.
        mean_auc: value.
        mean_fpr: mean fpr list.
        mean_tpr: mean tpr list.
        title: model_name.
        path: to load the data.
    """

    # Plot ROC AUC Curves
    for i in range(len(fpr)):
        ax.plot(fpr[i], tpr[i], lw = 3, alpha = 0.5,
                label='ROC fold %d (area = %0.2f)' % (i, auc_list[i]))

    # Plot diagonal
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance',
            alpha=.8)

    # Plot Mean ROC AUC
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % 
            (mean_auc, std_auc),
            lw=2, alpha=1)
    
    # Plot Confidence Interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', 
                    alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # Settings
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title='ROC Curve')
    ax.legend(loc="lower right")

    # Save Figure
    filename = 'pr_roc_' + str(title) + '.jpg'
    plt.savefig(os.path.join(path,filename), dpi = 100)

    plt.show()


def CV(classifier, X_dict, y , k, param_dict, param_title_dictionary, 
       model_name, path):

    """ 
    Calculate cross validation metrics and ROC curves.

    Args:
        classifier: model.
        X_dict: dataset dictionary.
        y: labels.
        k: number of splits.
        param_dict: parameter dictionary for the model.
        param_title_dictionary: model hyperparameter title (String).
        model_name: self explanatory.
        path: to load the data.

    Output:
        precision_dict: precision scores per experiment dictionary.
        recall_dict: recall scores per experiment dictionary.
        f1_dict: f1 scores per experiment dictionary.
        auc_dict: auc scores per experiment dictionary.
    """

    # Desired Metrics to include
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    auc_dict = {}


    # Creating Stratified Fold for integration with CV
    skf = StratifiedKFold(n_splits = k)

    for dataset_key in X_dict:

        fig, ax = plt.subplots(1,2,figsize = (20,10), dpi = 100)

        X = X_dict[dataset_key]

        # Set Parameters
        classifier.set_params(**param_dict[dataset_key])
        
        # Experiment Title
        title = model_name + '_' + dataset_key + '_' + \
                                            param_title_dictionary[dataset_key]

        fig.suptitle(title)

        # Desired Metrics to include
        precision_list = []
        recall_list = []
        f1_list = []
        auc_list = []

        # Necessary Lists
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        tprs = []

        # Positive Rates
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, (train, test) in enumerate(skf.split(X,y)):

            # Separate in train and test samples
            X_train = X.iloc[train,:]
            y_train = y.iloc[train]
            X_test = X.iloc[test,:]
            y_test = y.iloc[test]

            # Fitting the model
            classifier.fit(X_train,y_train)

            # Predict
            y_pred = classifier.predict(X_test)

            # Find Desired Metrics and add it to list
            precision = precision_score(y_test,y_pred)
            precision_list.append(precision)
            recall = recall_score(y_test,y_pred)
            recall_list.append(recall)
            f1 = f1_score(y_test,y_pred)
            f1_list.append(f1)

            # Score function
            y_score = classifier.predict_proba(X_test)[:,1]

            # Average Precision
            average_precision = average_precision_score(y_test, y_score)

            # Plot Prediction-Recall Curve
            disp = plot_precision_recall_curve(classifier, X_test, y_test,
                                               ax = ax[0],
                                               label = 'PR Fold '+str(i)+\
                                               ' AP: {0:0.2f}'\
                                               .format(average_precision))
            ax[0].legend(loc = 'best')
            ax[0].set_title('Precision Recall Curve')

            # ROC AUC Curve
            fpr[i], tpr[i], thresholds = roc_curve(y_test,y_score)
            auc_list.append(auc(fpr[i],tpr[i]))
            aux = np.interp(mean_fpr, fpr[i], tpr[i])
            mean_tpr += aux
            mean_tpr[0] = 0.0
            tprs.append(aux)

        # Building Mean ROC AUC Curve
        mean_tpr /= k
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        plotrocauc(auc_list,fpr,tpr,tprs,mean_auc,mean_fpr,mean_tpr,title,path,
                   fig, ax = ax[1])

        # Add lists to dictionaries
        precision_dict[dataset_key] = precision_list
        recall_dict[dataset_key] = recall_list
        f1_dict[dataset_key] = f1_list
        auc_dict[dataset_key] = auc_list

    return precision_dict, recall_dict, f1_dict, auc_dict

def hyperparametertunning(classifier, X_dict, y, param_grid, k, scoring):

    """
    Tunes the hyperparameters of a given classifier using grid search.

    Args:
        classifier: model used to perform predictions.
        X_dict: dataset dictionary.
        y: dependent variable.
        param_grid: hyperparameter grid.
        k: number of folds for cross-validation.
        scoring: reference metric to evaluate tuning.

    Outputs:
        param_dict: models optimal hyperparameter grid.
        param_title_dictionary:
    """

    # Best Parameter dictionary
    param_dict = {}

    # Parameter title dictionary
    param_title_dictionary = {}

    # Specify CV
    skf = StratifiedKFold(n_splits = k)

    # Test each dataset
    for dataset_key in X_dict:

        X_train = X_dict[dataset_key]

        clf = GridSearchCV(classifier, param_grid, cv  = skf, scoring = scoring)
        clf.fit(X_train, y)

        # Store parameters
        param_dict[dataset_key] = clf.best_estimator_.get_params()

        hyperparameter_title = ''
        for hyperparameter in param_grid.keys():
            hyperparameter_title += hyperparameter + '=' + \
                        str(param_dict[dataset_key][hyperparameter]) + '-'

        # Store titles
        param_title_dictionary[dataset_key] = hyperparameter_title[:-1]

    return param_dict, param_title_dictionary
