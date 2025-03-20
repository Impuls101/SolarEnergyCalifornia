#!/usr/bin/env python
# coding: utf-8

# ## Stacking Trial
# This notebook contains a Stacking trial which uses Hyperparameter optimization with Optuna and nested CV.

# In[5]:


import pandas as pd
#path = r"..\data\processed\\"
path = "~/dec24_bds_solar_energy/data/processed/" # TODO: Delete

X_train = pd.read_csv(path+'X_train_os_robust.csv')
X_test = pd.read_csv(path+'X_test_robust.csv')
y_train = pd.read_csv(path+'y_train_os.csv')
y_test = pd.read_csv(path+'y_test.csv')

print(X_train.shape,y_train.shape)
X_train.head(1)


# In[12]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score

from xgboost import XGBClassifier
#from tpot import TPOTClassifier

import optuna
#import optuna_distributed
from optuna.pruners import MedianPruner # to fasten Optuna

import shap
import warnings
warnings.filterwarnings("ignore")

# Define base models (level-0 models) and meta-learner (level-1 model)
# - Base models: RandomForest, SV, KNN and XGBoost; Meta-learner: LogisticRegression
base_models = [
    ("rf", RandomForestClassifier),
    ("svc", SVC),
    ("knn", KNeighborsClassifier),
    ("xgb", XGBClassifier)
    #,("tpot", TPOTClassifier(cv=3, max_time_mins=2, early_stop=3, n_jobs=1, random_state=888, verbosity=0)) # n_jobs=1 is necessary for the stacking classifiers it must be 1!
]

meta_model = LogisticRegression(random_state=888) # XGBClassifier(random_state=888) # RandomForestClassifier(random_state=888) 

# Inner Loop
def tune_base_model_hyperparameters(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> float:
    """
    Tune hyperparameters for base models using Optuna.

    This function is called during the inner loop of nested CV to tune hyperparameters for the
    base models. It suggests hyperparameters for each base model, trains a stacking classifier 
    with those hyperparameters, and evaluates its performance on the validation set.

    Args:
        trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.
        X_train (np.ndarray): Training features (inner loop training set).
        y_train (np.ndarray): Training labels (inner loop training set).
        X_val (np.ndarray): Validation features (inner loop validation set).
        y_val (np.ndarray): Validation labels (inner loop validation set).

    Returns:
        float: Validation accuracy for the current set of hyperparameters.
    """

    params = {
        "rf": {
            "n_estimators": trial.suggest_int("rf_n_estimators", 80, 1000)
            # ,rf_max_depth = trial.suggest_int("rf_max_depth", 10000, 100000)  # Maximum tree depth
        },
        "svc": {
            "C": trial.suggest_float("svc_C", 0.01, 100.0, log=True), # Regularization parameter (C)
            #"kernel": trial.suggest_categorical("svc_kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_float("svc_gamma", 0.001, 1.0, log=True) # Kernel coefficient (gamma)
            #, #degree = trial.suggest_int("svc_degree", 2, 5) if svc_kernel == "poly" else 3  # Degree of the polynomial kernel (for 'poly' kernel only)
        },
        "knn": {
            "n_neighbors": trial.suggest_int("knn_n_neighbors", 1, 30),
            "leaf_size": trial.suggest_int("knn_leaf_size", 5, 200)
            #, weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])  # Weight function
            #, algorithm = trial.suggest_categorical("knn_algorithm", ["auto", "ball_tree", "kd_tree", "brute"])  # Algorithm
        },
        "xgb": {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 5, 100), # Number of boosting rounds (can slow the runs down extremely)
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.001, 0.1)
            #, xgb_max_depth = trial.suggest_int("xgb_max_depth", 1000, 1002)  # Maximum tree depth
        }
    }
    #print("Optuna parameter suggestion is completed in this iteration.")
    # Update base models with the suggested hyperparameters (Each base model is reinitialized with the suggested hyperparameters)
    base_models_tuned = []
    for name, model_class in base_models:
        model_params = params[name]  # Get the parameters for this model        
        if name in ["rf", "svc", "xgb"]:  # Add random_state only for models that require it
            model_params["random_state"] = 888 # random_state is fixed for reproducibility
        base_models_tuned.append((name, model_class(**model_params)))  # Call the class constructor
    #print("Optuna base_models_tuned is completed in this iteration. Base_model_tuned:", base_models_tuned)
    # Initialize the stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models_tuned,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=888), # TODO: more splits!
        n_jobs=-1 # n_jobs=-1: Use all available CPU cores for parallel processing TODO: change to "= 1" if TPOT is also used (but then the run takes much longer!)!
    )
    # print("Optuna stacking_clf definition is completed in this iteration.")
    # Fit the stacking classifier on the inner loop training set
    stacking_clf.fit(X_train, y_train)
    print("Optuna stacking_clf.fit() is completed in this iteration.")
    # Predict on the inner loop validation set
    y_pred = stacking_clf.predict(X_val)

    # Return validation accuracy as the objective value for Optuna
    return accuracy_score(y_val, y_pred)


# Outer Loop:
def nested_cv_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 5, # default value used when not specified
    inner_splits: int = 5, # default value used when not specified
    n_trials: int = 50, # default value used when not specified
    random_state: int = 888
) -> dict:
    """
    Perform nested stratified k-fold CV for hyperparameter tuning and evaluation.

    Nested CV consists of two loops:
    - Outer loop: Splits the data into training and validation sets to evaluate generalization performance.
    - Inner loop: Performs hyperparameter tuning on the training set using cross-validation.

    This function ensures unbiased evaluation by keeping the outer validation set separate
    from the hyperparameter tuning process. StratifiedKFold is used to preserve class distribution
    in each fold, which is important for classification tasks.

    Args:
        X (np.ndarray): Features (training set).
        y (np.ndarray): Labels (training set).
        outer_splits (int): Number of outer folds (default: 5).
        inner_splits (int): Number of inner folds for hyperparameter tuning (default: 5).
        n_trials (int): Number of Optuna trials for hyperparameter tuning (default: 50).

    Returns:
        dict: Results including scores (accuracy for each outer fold), mean score,
              standard deviation, and best hyperparameters for each fold.
    """
    # Initialize the outer loop with StratifiedKFold
    outer_kf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    # Lists to store results
    scores, auc_scores, best_params_list, best_scores_list = [], [], [], [] # Accuracy, AUC for each outer fold as well as best hyperparameters and their scores for each outer fold
    
    # Convert y to numpy array if it's a DataFrame/Series
    if hasattr(y, 'values'):
        y = y.values.ravel()

    # Outer loop: Iterate over each fold
    for fold_idx, (train_idx, val_idx) in enumerate(outer_kf.split(X, y)):
        print(f"Outer fold {fold_idx + 1}/{outer_splits}")

        # Split the data into training and validation sets for the outer loop
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print("Start of Hyperparameter Tuning Iteration")
        # Define the objective function for Optuna (inner loop)        
        def objective(trial): # - This function will be called by Optuna to evaluate different hyperparameter sets
            return tune_base_model_hyperparameters(trial, X_train, y_train, X_val, y_val)

        # Perform hyperparameter tuning using Optuna
        pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=1) # pruner terminates trials that are performing poorly early, saving time
        # n_startup_trials: The first x trials will run completely without pruning; n_warmup_steps: Within each trial, the first y steps are considered warm-up and won't be pruned
        study = optuna.create_study(direction="maximize", pruner=pruner) # - direction="maximize": Maximize accuracy
        num_threads = 4 # on the VM we have 2 cores with 2 threads each core
        study.optimize(objective, n_trials=n_trials, n_jobs=num_threads) # - n_trials: Number of hyperparameter combinations to try

        # Get the best hyperparameters from the inner loop
        best_params = study.best_params
        best_score = study.best_value  # This gives you the best score associated with the parameters
        print("The best parameters of the n_trials before are:", best_params)
        best_params_list.append(best_params) # contains all inner loop's (which contains of multiple [n_trials] trials) best parameters which are returned later to the main function
        best_scores_list.append(best_score)
        print("best_params_list:", best_params_list)
        print("best_scores_list:", best_scores_list)
        print("End of Hyperparameter Tuning Iteration")

        # Train the final model with the best hyperparameters
        # - Reinitialize base models with the best hyperparameters
        base_models_final = [
            ("rf", RandomForestClassifier(
                n_estimators=best_params["rf_n_estimators"],
                #max_depth=best_params["rf_max_depth"],
                random_state=random_state
            )),
            ("svc", SVC(
                C=best_params["svc_C"],
                #kernel=best_params["svc_kernel"],
                gamma=best_params["svc_gamma"],
                #degree=best_params["svc_degree"],
                probability=True,  # Required for stacking
                random_state=random_state
            )),
            ("knn", KNeighborsClassifier(
                n_neighbors=best_params["knn_n_neighbors"],
                #weights=best_params["knn_weights"],
                #algorithm=best_params["knn_algorithm"],
                leaf_size=best_params["knn_leaf_size"]
            )),
            ("xgb", XGBClassifier(
                n_estimators=best_params["xgb_n_estimators"],
                #max_depth=best_params["xgb_max_depth"],
                learning_rate=best_params["xgb_learning_rate"],
                random_state=random_state
            ))
        ]

        # Initialize the stacking classifier with the best hyperparameters (inner CV for stacking)
        stacking_clf_final = StackingClassifier(
            estimators=base_models_final,
            final_estimator=meta_model,
            cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state),
            n_jobs=-1  # - n_jobs=-1: Use all available CPU cores for parallel processing TODO: change to "= 1" if TPOT is also used (but then the run takes much longer!)!
        )

        # Fit and evaluate on outer validation set
        stacking_clf_final.fit(X_train, y_train)
	    # Predict on the outer loop validation set
        y_pred = stacking_clf_final.predict(X_val)
        y_pred_proba = stacking_clf_final.predict_proba(X_val)[:, 1]
	    # Compute accuracy for the current outer fold
        scores.append(accuracy_score(y_val, y_pred))
        auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        print(f"Outer fold {fold_idx + 1} accuracy: {scores[-1]:.4f}, "
              f"ROC-AUC: {auc_scores[-1]:.4f}")

    return {
        "scores": scores,
        "auc_scores": auc_scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "mean_auc": np.mean(auc_scores),
        "std_auc": np.std(auc_scores),
        "best_params": best_params_list,
        "best_scores": best_scores_list
    }


def train_and_evaluate_stacking_model(X_train, X_test, y_train, y_test,
                                    outer_splits, inner_splits, n_trials,
                                    random_state):
    """
    Train and evaluate a stacking classifier using nested CV.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        outer_splits (int): Number of outer CV folds
        inner_splits (int): Number of inner CV folds
        n_trials (int): Number of Optuna trials for hyperparameter tuning
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Results including model performance metrics and trained model
    """

    # Perform nested CV (Outer Loop with Inner Loop within)
    print("Performing nested cross-validation...")
    cv_results = nested_cv_evaluation(X_train, y_train, outer_splits, inner_splits, n_trials, random_state)

    # Print CV results
    print("\nNested CV Results:")
    print(f"Mean accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    print(f"Mean ROC-AUC: {cv_results['mean_auc']:.4f} (+/- {cv_results['std_auc']:.4f})")
    
    # Train final model with best parameters
    print("\nTraining final model...")
    #best_params_final = cv_results["best_params"]#[-1]
    best_idx = cv_results["best_scores"].index(max(cv_results["best_scores"]))  # Index of best score
    best_params_final = cv_results["best_params"][best_idx]
    print("\nBest hyperparameters of all folds:", best_params_final) # print("Final stacking model parameters: ", final_stacking_clf.get_params())

    base_models_final = [
        ("rf", RandomForestClassifier(n_estimators=best_params_final["rf_n_estimators"],
                                    #max_depth=best_params_final["rf_max_depth"], 
                                    random_state=random_state)),
        ("svc", SVC(C=best_params_final["svc_C"],
                    #kernel=best_params_final["svc_kernel"],
                    gamma=best_params_final["svc_gamma"],
                    #degree=best_params_final["svc_degree"],
                    probability=True,  # Required for stacking
                    random_state=random_state)),
        ("knn", KNeighborsClassifier(
                    n_neighbors=best_params_final["knn_n_neighbors"],
                    #weights=best_params_final["knn_weights"],
                    #algorithm=best_params_final["knn_algorithm"],
                    leaf_size=best_params_final["knn_leaf_size"])),
        ("xgb", XGBClassifier(n_estimators=best_params_final["xgb_n_estimators"],
                             #max_depth=best_params_final["xgb_max_depth"],
                             learning_rate=best_params_final["xgb_learning_rate"], 
                             random_state=random_state))
    ]

    final_stacking_clf = StackingClassifier(
        estimators=base_models_final,
        final_estimator=LogisticRegression(random_state=random_state),
        cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state),
        n_jobs=-1 # TODO: change to "= 1" if TPOT is also used (but then the run takes much longer!)! (otherwise raises error with TPOT!)
    )
    final_stacking_clf.fit(X_train, y_train)

    # Final evaluation
    y_pred_test = final_stacking_clf.predict(X_test)
    y_pred_proba_test = final_stacking_clf.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    print(f"\nFinal test set - Accuracy: {test_accuracy:.4f}, ROC-AUC: {test_roc_auc:.4f}")
    
    # Generate confusion matrix
    print(confusion_matrix(y_test, y_pred_test))

    f1_score_test = f1_score(y_test, y_pred_test)
    y_pred_train = final_stacking_clf.predict(X_train)
    f1_score_train = f1_score(y_train, y_pred_train)
    print('F1-score for Test Set: ', f1_score_test)
    print('F1-score for Training Set: ', f1_score_train)

    # Classification report
    print("\nClassification Report for Test Set:")
    print(classification_report(y_test, y_pred_test)) 

    # ROC Curve Plot
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    ## SHAP Analysis
    print("\nComputing SHAP values...")
    # Summarize background data using k-means (e.g., K=1) to reduce computing time
    background_summary = shap.kmeans(X_train, 1)
    explainer = shap.KernelExplainer(final_stacking_clf.predict_proba, background_summary)

    # Compute SHAP values for the test set (X_test)
    shap_values = explainer.shap_values(X_test)

    # SHAP Values Distribution Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, max_display=10, show=True)
    ax.set_title('SHAP Values Distribution')
    plt.tight_layout()
    plt.show()

    return {
        "final_model": final_stacking_clf,
        "test_metrics": {
            "accuracy": test_accuracy,
            "roc_auc": test_roc_auc,
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True)
        },
        "cv_results": {
            "accuracy_scores": cv_results["scores"],
            "roc_auc_scores": cv_results["auc_scores"],
            "mean_accuracy": cv_results["mean_score"],
            "std_accuracy": cv_results["std_score"],
            "mean_roc_auc": cv_results["mean_auc"],
            "std_roc_auc": cv_results["std_auc"],
            "best_params": cv_results["best_params"]
        },
        #"shap_values": shap_values
    }, final_stacking_clf

results, final_stacking_clf = train_and_evaluate_stacking_model(X_train, X_test, y_train, y_test, 
                                                                outer_splits=5, inner_splits=2, n_trials=3, random_state=888)

