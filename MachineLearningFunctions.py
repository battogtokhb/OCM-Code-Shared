
# coding: utf-8

# In[3]:

from __future__ import division
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn.metrics import euclidean_distances
import pandas as pd
import numpy as np
import math as math
from sklearn.cross_validation import train_test_split
import sklearn
import pydot
from pylab import *
import csv
from sklearn import preprocessing
get_ipython().magic(u'matplotlib inline')

from itertools import cycle
from textwrap import wrap
import matplotlib.colors as colors
import matplotlib.cm as cmx
from collections import OrderedDict
from sklearn.cross_validation import KFold

import pylab


def probability_to_classification (probabilities, threshold=0.5):
    return [0 if x <= threshold else 1 for x in probabilities ]

def errors (name, prediction, target):

    cm = sklearn.metrics.confusion_matrix(np.asarray(prediction),np.asarray( target) )
    error = int(cm[0][1]) + int(cm[1][0])
    percentage = round(error / len(prediction), 3)
    return (name, percentage)




def RFR (setting, train, train_target, predict, test_target, verbose=False, error= False, reg=False):
    name = "RF, min_weight_fraction_leaf= " + str(setting)
    rfr = RandomForestRegressor(n_estimators =100, min_weight_fraction_leaf=setting)
    rfr = rfr.fit(train, train_target)
    rfr_prediction = rfr.predict(predict)
    classification_based_prob = []
    for x in rfr_prediction:
        if x <= 0.5:
            classification_based_prob.append(0)
        else:
            classification_based_prob.append(1)
                
    if verbose:
        print pd.crosstab(pd.Series(test_target), pd.Series(classification_based_prob), rownames=['True'], colnames=['Predicted'], margins=True), " RF, min_weight_fraction_leaf= " + str(setting) , "Regression"
    if error:
        return errors(name, classification_based_prob, test_target)
    if reg:
        return name, rfr_prediction
    else:
        return classification_based_prob


def KNNR (setting, train,  train_target,predict,test_target,  verbose=False, error= False, reg=False):
    name = "KNN, K= " + str(setting)
    knnr = KNeighborsRegressor(n_neighbors=setting)
    knnr = knnr.fit(train, train_target)
    knnr_prediction = knnr.predict(predict)
    classification_based_prob = []
    for x in knnr_prediction:
        if x <= 0.5:
            classification_based_prob.append(0)
        else:
            classification_based_prob.append(1)
    if verbose:
        print pd.crosstab(pd.Series(test_target), pd.Series(classification_based_prob), rownames=['True'], colnames=['Predicted'], margins=True), " KNN, K= " + str(setting) , "Regression"
    if error:
        return errors(name, classification_based_prob, test_target)
    if reg:
        return name, knnr_prediction
    else:
        return classification_based_prob
    

def SVR (setting, train,  train_target,predict,test_target,  verbose=False, error= False,reg=False):
    name = "SV, kernal= " + str(setting)
    svr = svm.SVR(kernel=setting)
    svr = svr.fit(train, train_target)
    svr_prediction = svr.predict(predict)
    classification_based_prob = []
    for x in svr_prediction:
        if x <= 0.5:
            classification_based_prob.append(0)
        else:
            classification_based_prob.append(1)
                
    if verbose:
        print pd.crosstab(pd.Series(test_target), pd.Series(classification_based_prob), rownames=['True'], colnames=['Predicted'], margins=True), "SVM, Kernal= " + str(setting) , "Regression"

    if error:
        return errors( name, classification_based_prob, test_target)

    if reg:
            return name, svr_prediction
    else:
        return classification_based_prob
    
    
def KNNC (setting, train,  train_target,predict, test_target, verbose=False, error= False, reg=False):
    knnc = KNeighborsClassifier(n_neighbors=setting)
    knnc = knnc.fit(train, train_target)
    knnc_prediction = knnc.predict(predict)

    if verbose:
        print pd.crosstab(pd.Series(test_target), pd.Series(knnc_prediction), rownames=['True'], colnames=['Predicted'], margins=True), " KNN, K= " + str(setting) , "Classification"

    if error:
        return errors("KNN, K= " + str(setting) , knnc_prediction, test_target)

    else:
        return knnc_prediction.round(2)


def RFC(setting, train, train_target, predict,test_target,  verbose=False, error= False, reg=False):
    rfc = RandomForestClassifier(n_estimators= 100, min_weight_fraction_leaf=setting)
    rfc = rfc.fit(train, train_target)
    rfc_prediction = rfc.predict(predict)

    if verbose:
        print pd.crosstab(pd.Series(test_target), pd.Series(rfc_prediction), rownames=['True'], colnames=['Predicted'], margins=True), " RF, min_weight_fraction_leaf= " + str(setting) , "Classification"
    if error:
        return errors("RF, min_weight_fraction_leaf= " + str(setting), rfc_prediction, test_target)

    else:
        return rfc_prediction.round(2)


def SVC (setting, train,  train_target,predict,test_target,  verbose=False, error= False,reg=False):
    svc = svm.SVC(kernel=setting)
    svc = svc.fit(train, train_target)
    svc_prediction = svc.predict(predict)
    if verbose:
        print pd.crosstab(pd.Series(test_target), pd.Series(svc_prediction), rownames=['True'], colnames=['Predicted'], margins=True), "SVM, Kernal= " + str(setting) , "Classification"

    if error:
        return errors("SV, kernel= " + str(setting), svc_prediction, test_target)

    else:
        return svc_prediction



# In[4]:

def OPTIMAL_CLASSIFIER(train_features, train_target, test_features, test_target, machines,  SYN=True, ERROR=True, CV=False, VERBOSE=False, *args):
    mse_dict = {}
    errors_dict = {}
    
    functions_dict = {'KNNC': KNNC, 'RFC': RFC, 'SVC': SVC, 'KNNR': KNNR, 'RFR': RFR, 'SVR': SVR}
    
    empty_cell_count = 0
    regular_cell_count = 0
    
    
    #Generate synthetic features for training and testing -> final = original features + synthetic 

    final_train_features = train_features
    if SYN:
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
            #print functions_dict[machine_name](machine_setting, train_features )
                if VERBOSE:
                    print machine_name, machine_setting, "train sf"
                final_train_features =  np.column_stack( (final_train_features, np.reshape(functions_dict[machine_name](machine_setting, train_features,  train_target,train_features,test_target ), len(train_features ), 1))  )

        

    final_test_features = test_features
    if SYN:
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
                    if VERBOSE:
                        print machine_name, machine_setting, "test sf"
                    final_test_features =  np.column_stack( (final_test_features, np.reshape(functions_dict[machine_name](machine_setting, train_features, train_target,test_features,test_target), len(test_features ), 1))  )


    #Generate optimal crowd classifier
    def closest_vectors (vector, list_of_vector):
        distances = {}
        for some_vector in list_of_vector:
            distance = euclidean_distances(np.array(vector).reshape(1,-1), np.array(some_vector).reshape(1,-1))[0][0]
            if distance not in distances.keys():
                distances[distance] = []
                distances[distance].append(some_vector)
            else:
                distances[distance].append(some_vector)

        distances_keys = sorted(distances.keys())
        return distances[distances_keys[0]]


    total_train_machines = []
    for machine_name, machine_settings in machines:
        for machine_setting in machine_settings:
            if VERBOSE:
                print machine_name, machine_setting, "train dict"
            total_train_machines.append( functions_dict[machine_name](machine_setting, final_train_features,  train_target,final_train_features,test_target ) )


#print
    
    #Generate dictionary of classification outcomes vs. real outcomes for training data
    
    training_outcomes = {}
    i = 0
    while i <= len(train_features) -1:
        arrangement = ""
        for machine in total_train_machines:
            arrangement +=  str(int(machine[i]))
        if tuple(map(int, arrangement)) not in training_outcomes.keys():
            training_outcomes[tuple(map(int, arrangement))] = []
            training_outcomes[tuple(map(int, arrangement))].append(int(train_target[i]) )
        else:
            training_outcomes[tuple(map(int, arrangement))].append(int(train_target[i]) )
        i+=1

    training_cell_list = []
    for key in training_outcomes.keys():
        training_cell_list.append(len(training_outcomes[key]) )
    
    
    #print "TRAINING CELL OUTCOMES"
    training_cell_dict = {}
    for i in range(0, len(training_cell_list) ):
        training_cell_dict[i] =  training_cell_list[i]
        
    #print training_cell_dict

    total_test_machines = []
    for machine_name, machine_settings in machines:
        for machine_setting in machine_settings:
            if VERBOSE:
                print machine_name, machine_setting, "test dict"
            total_test_machines.append( functions_dict[machine_name](machine_setting, final_train_features,  train_target,final_test_features,test_target ) )



    #look up classification outcomes for testing data, average against real outcomes

    total_test_predictions = []
    p = 0

    while p <= len(test_features) -1:
        arrangement = ""
        for machine in total_test_machines:
            arrangement +=  str(int(machine[p]))
#print arrangement, training_outcomes.keys()
        list_of_keys = closest_vectors(tuple(map(int, arrangement)), training_outcomes.keys())
        closest_data = [ training_outcomes[key] for key in list_of_keys]
        average = [sum(e)/len(e) for e in zip(*closest_data)]
        total_test_predictions.append(np.mean(average))
#sprint average,np.mean(average)
        p +=1 
        
    #print total_test_predictions
    
    final_test_predictions = []
    for prediction in total_test_predictions:
        if prediction <= 0.5:
            final_test_predictions.append(0)
        else:
            final_test_predictions.append(1)

    
    #print final_test_predictions
    if VERBOSE and ERROR:
        print pd.crosstab(pd.Series(test_target), pd.Series(final_test_predictions), rownames=['True'], colnames=['Predicted'], margins=True), "Optimal Crowd Classifier"
        for machine_name, machine_settings in machines:
                for machine_setting in machine_settings:
                    error = functions_dict[machine_name](machine_setting, final_train_features,  train_target,final_test_features,test_target, error=ERROR, verbose=VERBOSE)
                    #print "ERROR", error
                    errors_dict[error[0]] = error[1]
    elif VERBOSE:
        print pd.crosstab(pd.Series(test_target), pd.Series(final_test_predictions), rownames=['True'], colnames=['Predicted'], margins=True), "Optimal Crowd Classifier"
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
                functions_dict[machine_name](machine_setting, final_train_features,  train_target,final_test_features,test_target, verbose=VERBOSE, error=ERROR)
    elif ERROR:
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
                error = functions_dict[machine_name](machine_setting, final_train_features,  train_target,final_test_features,test_target, verbose=VERBOSE, error=ERROR)
                errors_dict[error[0]] = error[1]
    else:
        pass


    if CV:
        return 1 - sklearn.metrics.accuracy_score(test_target,final_test_predictions)

    if SYN:
        return errors("Optimal Crowd Classifier " + "SYN=" + str(SYN), final_test_predictions, test_target), final_test_predictions


    if ERROR:
        error = errors("Optimal Crowd Classifier " + "SYN=" + str(SYN), final_test_predictions, test_target)
        errors_dict[error[0]] = error[1]

    case_prob = {}
    control_prob = {}

    

#print "# empty cells used ", empty_cell_count, "# regular cells used ", regular_cell_count
    if REG:
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
                name, prediction = functions_dict[machine_name](machine_setting, final_train_features,  train_target,final_test_features,test_target, reg=True )
                case_prob[name] = prediction[case_index] 
                control_prob[name] =  prediction[control_index] 
        control_prob ["Optimal Crowd Classifier " + "SYN=" + str(SYN)] = total_test_predictions[control_index]
        case_prob ["Optimal Crowd Classifier " + "SYN=" + str(SYN)] = total_test_predictions[case_index]
        return errors_dict, case_prob, control_prob

    return errors_dict




def OPTIMAL_REGRESSOR(train_features, train_target, test_features, test_target, machines, SYN=True, VERBOSE=False, CV=False):
    mse_dict = {}
    errors_dict = {}
    functions_dict = {'KNNR': KNNR, 'RFR': RFR, 'SVR': SVR}
   
    final_train_features = train_features
    if SYN:
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
            #print functions_dict[machine_name](machine_setting, train_features )
                if VERBOSE:
                    print machine_name, machine_setting, "train sf"
                final_train_features =  np.column_stack( (final_train_features, np.reshape(functions_dict[machine_name](machine_setting, train_features,  train_target,train_features,test_target ), len(train_features ), 1))  )


    final_test_features = test_features
    if SYN:
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
                if VERBOSE:
                    print machine_name, machine_setting, "test sf"
                final_test_features =  np.column_stack( (final_test_features, np.reshape(functions_dict[machine_name](machine_setting, train_features, train_target,test_features,test_target), len(test_features ), 1))  )

    print "final_test_features: ", np.array(final_test_features).shape, "final_train_features: ", np.array(final_train_features).shape

#Generate optimal crowd classifier
    train_dict = {}
    test_dict = {}
    for machine_name, machine_settings in machines:
        for machine_setting in machine_settings:
            train_entry =functions_dict[machine_name](machine_setting, final_train_features, train_target,final_train_features,test_target, reg=True )[1]
            
            train_array = zip(train_entry, train_target)
            for prediction, target in train_array:
                if prediction not in train_dict.keys():
                    train_dict[prediction] = []
                    train_dict[prediction].append(target)
                else:
                    train_dict[prediction].append(target)
            
            test_entry =functions_dict[machine_name](machine_setting, final_train_features, train_target,final_test_features,test_target, reg=True )[1]
            test_dict[machine_name + str(machine_setting)] = test_entry.tolist()

    prediction_list = []
    for key in test_dict.keys():
        machine_prediction = []
        for test_prediction in test_dict[key]:
            train_key = min(train_dict.keys(), key=lambda x:abs(x-test_prediction))
            machine_prediction.append(np.mean(train_dict[train_key]))



    prediction_list.append(machine_prediction)

    final_prediction = [sum(x)/len(x) for x in zip(*prediction_list)]
    final_prediction_classification = []
    for x in final_prediction:
        if x <= 0.5:
            final_prediction_classification.append(0)
        else:
            final_prediction_classification.append(1)


    print errors("Optimal Crowd Regressor" + " SYN=" + str(SYN), final_prediction_classification, test_target)
    
    if CV:
        return 1-sklearn.metrics.accuracy_score(test_target,final_prediction_classification)

    return errors("Optimal Crowd Regressor" + " SYN=" + str(SYN), final_prediction_classification, test_target), final_prediction


def OPTIMAL_REGRESSOR_ALPHA(train_features, train_target, test_features, test_target, machines, ALPHA=.05, SYN=True, VERBOSE=False, CV=False):
    mse_dict = {}
    errors_dict = {}
    functions_dict = {'KNNR': KNNR, 'RFR': RFR, 'SVR': SVR}
    
    final_train_features = train_features
    if SYN:
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
                #print functions_dict[machine_name](machine_setting, train_features )
                if VERBOSE:
                    print machine_name, machine_setting, "train sf"
                final_train_features =  np.column_stack( (final_train_features, np.reshape(functions_dict[machine_name](machine_setting, train_features,  train_target,train_features,test_target ), len(train_features ), 1))  )


    final_test_features = test_features
    if SYN:
        for machine_name, machine_settings in machines:
            for machine_setting in machine_settings:
                if VERBOSE:
                    print machine_name, machine_setting, "test sf"
                final_test_features =  np.column_stack( (final_test_features, np.reshape(functions_dict[machine_name](machine_setting, train_features, train_target,test_features,test_target), len(test_features ), 1))  )

    print "final_test_features: ", np.array(final_test_features).shape, "final_train_features: ", np.array(final_train_features).shape
    
    #Generate optimal crowd classifier
    train_dict = {}
    test_dict = {}
    for machine_name, machine_settings in machines:
        for machine_setting in machine_settings:
            train_entry =functions_dict[machine_name](machine_setting, final_train_features, train_target,final_train_features,test_target, reg=True )[1]
            train_array = zip(train_entry, train_target)
            for prediction, target in train_array:
                if prediction not in train_dict.keys():
                    train_dict[prediction] = []
                    train_dict[prediction].append(target)
                else:
                    train_dict[prediction].append(target)
            
            test_entry =functions_dict[machine_name](machine_setting, final_train_features, train_target,final_test_features,test_target, reg=True )[1]
            test_dict[machine_name + str(machine_setting)] = test_entry.tolist()
                
    def closest_vectors (vector, list_of_vector, alpha=ALPHA):
            distances = {}
            for some_vector in list_of_vector:
                distance = euclidean_distances(np.array(vector).reshape(1,-1), np.array(some_vector).reshape(1,-1))[0][0]
                if distance <= alpha and distance not in distances.keys():
                    distances[distance] = None
                    distances[distance] = some_vector
                elif distance <= alpha:
                    distances[distance] = some_vector
                else:
                    pass
            values = distances.values()
            return values
    count = 0
    prediction_list = []
    for key in test_dict.keys():
      machine_prediction = []
      for test_prediction in test_dict[key]:
             list_of_keys = closest_vectors( test_prediction, train_dict.keys())
             closest_data = [ train_dict[x] for x in list_of_keys]
             average = [sum(e)/len(e) for e in zip(*closest_data)]
             machine_prediction.append(average)
             count +=1
             print key, count




    prediction_list.append(machine_prediction)
    
    final_prediction = [sum(x)/len(x) for x in zip(*prediction_list)]
    final_prediction_classification = []
    for x in final_prediction:
        if x <= 0.5:
            final_prediction_classification.append(0)
        else:
            final_prediction_classification.append(1)


    print errors("Optimal Crowd Regressor" + " SYN=" + str(SYN), final_prediction_classification, test_target)
    
    if CV:
        return 1-sklearn.metrics.accuracy_score(test_target,final_prediction_classification)

    return errors("Optimal Crowd Regressor" + " SYN=" + str(SYN), final_prediction_classification, test_target), final_prediction





def OPTIMAL_CV(features, target, machines, classifier=True, VERBOSE=False, SYN=False):
    
    kf = KFold(len(features), n_folds=5)
    error_list = []
    for train_index, test_index in kf:
        train_features, train_target =  np.array(features)[train_index], np.array(target)[train_index]
        test_features, test_target = np.array(features)[test_index], np.array(target)[test_index]
        if classifier:
            error_list.append(OPTIMAL_CLASSIFIER(train_features, train_target, test_features, test_target, machines, CV=True, SYN=SYN))
        else:
            error_list.append(OPTIMAL_REGRESSOR(train_features, train_target, test_features, test_target, machines, CV=True, SYN=SYN))
    if VERBOSE:
        print error_list

    class_name = 'Regressor'
    if classifier:
        class_name = 'Classifier'

    name = "Optimal Crowd " + class_name + " SYN=" + str(SYN)
    mean = np.mean(error_list)
    difference = [(x-mean)**2 for x in error_list]
    sum_difference = sum(difference)
    sum_difference = sum_difference / (5-1)
    variance = sqrt(sum_difference)
    cv_dict = {}
    cv_dict[name] = [np.mean(error_list), variance]
    return cv_dict




def cross_val(features, target, machines, VERBOSE=False):
    
    #def RFR (setting, train, train_target, predict, test_target, verbose=False, error= False, reg=False):

    functions_dict = {'KNNC': KNNC, 'RFC': RFC, 'SVC': SVC, 'KNNR': KNNR, 'RFR': RFR, 'SVR': SVR}
    settings_dict = {'KNNC': 'k', 'RFC': 'min_weight_fraction_leaf', 'SVC': 'kernel', 'KNNR': 'k', 'RFR': 'min_weight_fraction_leaf', 'SVR': 'kernel'}
    
    kf = KFold(len(features), n_folds=5)
    cross_val_dict = {}
    for machine_name, machine_settings in machines:
        for machine_setting in machine_settings:
            error_list = []
            
            for train_index, test_index in kf:
                train_features, train_target = np.array(features)[train_index], np.array(target)[train_index]
                test_features, test_target = np.array(features)[test_index], np.array(target)[test_index]
                prediction = functions_dict[machine_name](machine_setting, train_features, train_target, test_features, test_target)
                #print pd.crosstab(pd.Series(test_target), pd.Series(prediction), rownames=['True'], colnames=['Predicted'], margins=True), machine_name + str(machine_setting)
                error_list.append(1-sklearn.metrics.accuracy_score(test_target, prediction ) )
            
           
            
            mean = np.mean(error_list)
            difference = [(x-mean)**2 for x in error_list]
            sum_difference = sum(difference)
            sum_difference = sum_difference / (5-1)
            variance = sqrt(sum_difference)
            if VERBOSE:
                print machine_name, machine_setting, 'VALUES: ', error_list, 'VARIANCE: ', variance
            cross_val_dict[machine_name[:-1] + ", " + settings_dict[machine_name] + '= ' + str(machine_setting)] = [np.mean(error_list), variance]


    return cross_val_dict












