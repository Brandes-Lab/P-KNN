import numpy as np
import pandas as pd



def get_score_rank(test_array, calibration_array):
    """
    Calculates the rank scores of each element in the test array relative to the corresponding column in the calibration array.
    For each column, the function computes the rank of each non-NaN value in the test array based on its position within the sorted calibration array.
    If a value is NaN, its rank is set to NaN. The rank is computed as the average of the left and right insertion indices (1-based) in the sorted calibration array.
    Parameters:
        test_array (np.ndarray): 2D array of test values, shape (n_samples, n_features).
        calibration_array (np.ndarray): 2D array of calibration values, shape (n_calibration_samples, n_features).
    Returns:
        np.ndarray: 2D array of rank scores, shape (n_samples, n_features), where each entry is the rank of the corresponding test value or NaN if the test value is NaN.
    """
    rank_scores = []
    for col_idx in range(calibration_array.shape[1]):  # deal with each column
        calibration_sorted = np.sort(calibration_array[:, col_idx]) 

        col_rank = np.full(test_array.shape[0], np.nan)
        
        # Select index of not nan element, get the rank of the non-nan elements in test_array
        non_nan_indices = ~np.isnan(test_array[:, col_idx])
        if non_nan_indices.any():
            col_rank[non_nan_indices] = (
                (np.searchsorted(calibration_sorted, test_array[non_nan_indices, col_idx], side="left") +
                 np.searchsorted(calibration_sorted, test_array[non_nan_indices, col_idx], side="right") + 1) / 2
            )
        
        rank_scores.append(col_rank)
    
    return np.column_stack(rank_scores)



def get_rank(calibration_array, test_array, regularization_array):
    """
    Calculates the rank scores for calibration, test, and regularization arrays against the calibration array.
    Args:
        calibration_array (np.ndarray): 2D array of calibration data, shape (n_calibration_samples, n_features).
        test_array (np.ndarray): 2D array of test values, shape (n_samples, n_features).
        regularization_array (np.ndarray): 2D array of regularization values, shape (n_regularization_samples, n_features).
    Returns:
        tuple: A tuple containing three elements:
            - calibration_rank: Rank scores of calibration_array against itself.
            - test_rank: Rank scores of test_array against calibration_array.
            - regularization_rank: Rank scores of regularization_array against calibration_array.
    """
    calibration_rank = get_score_rank(calibration_array, calibration_array)
    test_rank = get_score_rank(test_array, calibration_array)
    regularization_rank = get_score_rank(regularization_array, calibration_array)

    return calibration_rank, test_rank, regularization_rank



def get_z(calibration_array, test_array, regularization_array):
    """
    Standardizes (z-score normalizes) the columns of the input arrays using the mean and standard deviation 
    computed from the non-NaN values of each column in the calibration_array.
    For each column:
        - Computes the mean and standard deviation from non-NaN values in calibration_array.
        - Applies z-score normalization to calibration_array, test_array, and regularization_array using the computed mean and std.
        - Preserves NaN values in the output arrays.
        - If the standard deviation is zero, uses 1 to avoid division by zero.
    Parameters:
        calibration_array (np.ndarray): 2D array of calibration data, shape (n_calibration_samples, n_features).
        test_array (np.ndarray): 2D array of test values, shape (n_samples, n_features).
        regularization_array (np.ndarray): 2D array of regularization values, shape (n_regularization_samples, n_features).
    Returns:
        tuple: A tuple containing three np.ndarrays (calibration_z, test_z, regularization_z), 
               which are the z-score normalized versions of the input arrays.
    """
    calibration_z = np.zeros_like(calibration_array)
    test_z = np.zeros_like(test_array)
    regularization_z = np.zeros_like(regularization_array)

    for col in range(calibration_array.shape[1]):
        # calculate mean and std with not nan value in calibration_array
        calibration_column = calibration_array[:, col]
        valid_values = calibration_column[~np.isnan(calibration_column)]
        
        mean = np.mean(valid_values)
        std = np.std(valid_values)

        if std == 0: std = 1 # avoid error if std is 0

        # do z transform
        calibration_z[:, col] = np.where(~np.isnan(calibration_array[:, col]), 
                                   (calibration_array[:, col] - mean) / std, 
                                   np.nan)
        test_z[:, col] = np.where(~np.isnan(test_array[:, col]), 
                                  (test_array[:, col] - mean) / std, 
                                  np.nan)
        regularization_z[:, col] = np.where(~np.isnan(regularization_array[:, col]), 
                                    (regularization_array[:, col] - mean) / std, 
                                    np.nan)
        
    return calibration_z, test_z, regularization_z



def KNN_impute(calibration_array, test_array, regularization_array, n_neighbors = 100):
    """
    KNN_impute(calibration_array, test_array, regularization_array, n_neighbors=100)
    Builds a K-Nearest Neighbors (KNN) imputer using the sklearn KNNImputer class to handle missing values 
    in the provided datasets. The imputer is calibrationed on the `calibration_array` and then applied to impute 
    missing values in `calibration_array`, `test_array`, and `regularization_array`.
    Parameters:
        calibration_array (np.ndarray): 2D array of calibration data, shape (n_calibration_samples, n_features).
        test_array (np.ndarray): 2D array of test values, shape (n_samples, n_features).
        regularization_array (np.ndarray): 2D array of regularization values, shape (n_regularization_samples, n_features).
        n_neighbors (int, optional): The number of neighbors to use for imputation. Defaults to 100 
                                     (same as calibration window size).
    Returns:
        tuple: A tuple containing three 2D arrays with the same shapes as `calibration_array`, `test_array`, 
               and `regularization_array`, respectively, but with missing values imputed.
               - filled_calibration (numpy.ndarray): Imputed version of `calibration_array`.
               - filled_test (numpy.ndarray): Imputed version of `test_array`.
               - filled_regularization (numpy.ndarray): Imputed version of `regularization_array`.
    Example:
        >>> filled_calibration, filled_test, filled_regularization = KNN_impute(calibration_array, test_array, regularization_array, n_neighbors=100)
    """
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors = n_neighbors, weights="uniform")
    filled_calibration = imputer.fit_transform(calibration_array)
    filled_test = imputer.transform(test_array)
    filled_regularization = imputer.transform(regularization_array)

    return filled_calibration, filled_test, filled_regularization



def mutual_information_scaling(calibration_array, test_array, regularization_array, calibration_label):
    """
    Scales input arrays by feature-wise mutual information with respect to calibration labels.
    This function computes the mutual information between each feature in the calibration array and the calibration labels.
    It then normalizes the mutual information values to obtain scaling factors, which are used to scale the features of
    the calibration, test, and regularization arrays. The scaled arrays and the scaling factors are returned.
    Parameters:
        calibration_array (np.ndarray): 2D array of calibration data, shape (n_calibration_samples, n_features).
        test_array (np.ndarray): 2D array of test values, shape (n_samples, n_features).
        regularization_array (np.ndarray): 2D array of regularization values, shape (n_regularization_samples, n_features).
        calibration_label (np.ndarray or list): Labels corresponding to the calibration array (shape: [n_samples]).
    Returns: 
        calibration_mi_normalized : np.ndarray
            Calibration array scaled by mutual information-based scaling factors.
        test_mi_normalized : np.ndarray
            Test array scaled by mutual information-based scaling factors.
        regularization_mi_normalized : np.ndarray
            Regularization array scaled by mutual information-based scaling factors.
        scaling_factor : np.ndarray
            Array of scaling factors derived from normalized mutual information for each feature.
    """
    from sklearn.feature_selection import mutual_info_classif
    mutual_information = mutual_info_classif(calibration_array, calibration_label)
    scaling_factor = mutual_information / np.max(mutual_information)
    calibration_mi_normalized = calibration_array * scaling_factor 
    test_mi_normalized = test_array * scaling_factor 
    regularization_mi_normalized = regularization_array * scaling_factor     

    return calibration_mi_normalized, test_mi_normalized, regularization_mi_normalized, scaling_factor



def get_weight(labels: np.ndarray, Pprior: float):
    """
    Calculate the weight for benign labels based on the prior probability.
    Parameters:
        labels (numpy.ndarray or array-like): Array of binary labels where 1 indicates benign and 0 indicates pathogenic.
        Pprior (float): Prior probability of being benign (between 0 and 1, exclusive).
    Returns:
        float: Weight for benign labels, adjusted by prior probability.
    Notes:
        - The function computes a weight for benign labels based on the prior probability.
    """
    w_benign = (1 - Pprior) * sum(labels == 1) / (sum(labels == 0) * Pprior)
    return w_benign



def get_logbase(Pprior: float, Ppost_thresholds = [0.99, 0.9, 0.1, 0.01], ACMGscore_thresholds = [10, 6, -1, -4]):
    """
    Calculate the maximum logarithmic base for likelihood ratios given prior probabilities,
    posterior probability thresholds, and ACMG score thresholds.
    Args:
        Pprior (float): The prior probability, a value between 0 and 1.
        Ppost_thresholds (list of float, optional): A list of posterior probability thresholds.
            Defaults to [0.99, 0.9, 0.1, 0.01].
        ACMGscore_thresholds (list of int, optional): A list of ACMG score thresholds corresponding
            to the posterior probability thresholds. Defaults to [10, 6, -1, -4].
    Returns:
        float: The maximum logarithmic base calculated from the likelihood ratios.
    """
    Prior_odds = Pprior / (1 - Pprior)
    logbase_list = []
    for Ppost, ACMGscore in zip(Ppost_thresholds, ACMGscore_thresholds): #(Posterior probability, ACMG score threshold)
        Posterior_odds = Ppost / (1 - Ppost)
        Likelihood_ratio = Posterior_odds / Prior_odds
        logbase = Likelihood_ratio**(8/ACMGscore)
        logbase_list.append(logbase)

    return max(logbase_list)



def get_calibration_percentage(calibration_array, test_row, regularization_array, calibration_label, w_calibration, 
                               n_calibration_in_window = 100, frac_regularization_in_window=0.03):
    """
    Calculate the pathogenic percentage for a given test row based on its proximity 
    to calibrationing data and regularization data within a specified distance window.
    Parameters:
    -----------
    test_row : numpy.ndarray
        A single data point (row) from the test dataset.
    calibration_array : numpy.ndarray
        Array of calibrationing data points.
    regularization_array : numpy.ndarray
        Array of regularization data points.
    calibration_label : numpy.ndarray
        Array of labels corresponding to the calibrationing data points (1 for pathogenic, 0 for benign).
    w_calibration : float
        Weight assigned to benign samples in the calculation of the pathogenic percentage.
    n_calibration_in_window : int, optional
        Number of calibrationing samples to include within the distance window (default is 100).
    frac_regularization_in_window : float, optional
        Fraction of regularization samples to include within the distance window (default is 0.03).
    Returns:
    --------
    float
        The pathogenic percentage calculated as the ratio of pathogenic samples 
        to the weighted sum of benign and pathogenic samples within the distance window.
    """
    # Calculate distances from the test row to all calibrationing and regularization samples
    calibration_distances = np.linalg.norm(calibration_array - test_row, axis=1)
    regularization_distances = np.linalg.norm(regularization_array - test_row, axis=1)
    
    # determine the distance limit for the calibrationing and regularization samples
    n_regularization_in_window = int(np.ceil(regularization_array.shape[0] * frac_regularization_in_window))   
    distance_limit = max(np.sort(calibration_distances)[n_calibration_in_window], np.sort(regularization_distances)[n_regularization_in_window])
    
    calibration_index = np.where(calibration_distances <= distance_limit)[0]
    selected_labels = calibration_label[calibration_index]  
    pathogenic_count = sum(selected_labels)
    benign_count = len(selected_labels)-pathogenic_count
    pathogenic_percentage = pathogenic_count/(w_calibration*benign_count+pathogenic_count)

    return pathogenic_percentage



def get_KNN_score(calibration_array, test_array, regularization_array, 
                  calibration_label, w_calibration, 
                  n_calibration_in_window=100, frac_regularization_in_window=0.03, 
                  normalization='rank', impute=True, mi_scaling=True):
    """
    Computes KNN-based scores for test samples using a calibrationing set and a reference set (regularization).
    This function normalizes the input arrays, optionally imputes missing values, applies mutual information scaling,
    and then calculates a KNN-based calibration score for each test sample.
    Args:
        calibration_array (np.ndarray): Feature array for calibrationing samples (shape: n_calibration x n_features).
        test_array (np.ndarray): Feature array for test samples (shape: n_test x n_features).
        regularization_array (np.ndarray): Feature array for regularization/reference samples (shape: n_regularization x n_features).
        calibration_label (np.ndarray): Labels for calibrationing samples (shape: n_calibration,).
        w_calibration (np.ndarray): Sample weights for calibrationing samples (shape: n_calibration,).
        n_calibration_in_window (int, optional): Number of nearest neighbors from the calibrationing set to consider. Default is 100.
        frac_regularization_in_window (float, optional): Fraction of regularization samples to include in the window. Default is 0.03.
        normalization (str, optional): Normalization method to use ('rank', 'z', or 'minmax'). Default is 'rank'.
        impute (bool, optional): Whether to impute missing values. Default is True.
        mi_scaling (bool, optional): Whether to apply mutual information scaling. Default is True.
    Returns:
        np.ndarray: Array of KNN-based scores for each test sample (shape: n_test,).
    """
    if normalization == 'rank':
        calibration_array, test_array, regularization_array = get_rank(calibration_array, test_array, regularization_array)
    elif normalization == 'z':
        calibration_array, test_array, regularization_array = get_z(calibration_array, test_array, regularization_array)

    if not impute:
        valid_calibration_idx = np.all(~np.isnan(calibration_array), axis=1)
        calibration_array = calibration_array[valid_calibration_idx]
        calibration_label = calibration_label[valid_calibration_idx]

    calibration_array, test_array, regularization_array = KNN_impute(calibration_array, test_array, regularization_array, n_neighbors = n_calibration_in_window)
        
    if mi_scaling:
        calibration_array, test_array, regularization_array, _ = mutual_information_scaling(calibration_array, test_array, regularization_array, calibration_label)

    knn_score = np.array([get_calibration_percentage(calibration_array, row, regularization_array, calibration_label, w_calibration, 
                                                     n_calibration_in_window = n_calibration_in_window, frac_regularization_in_window=frac_regularization_in_window) 
                          for row in test_array])

    return knn_score



def get_one_bootstrap_KNN_score(calibration_array, test_array, regularization_array, calibration_label, w_calibration, 
                                n_calibration_in_window=100, frac_regularization_in_window=0.03, 
                                normalization='rank', impute=True, mi_scaling=True, seed=None):
    """
    Performs a single bootstrap resampling of the calibrationing data and computes the KNN score.
    This function generates a bootstrap sample from the calibrationing data, applies the KNN scoring function
    using the resampled data, and returns the resulting score. It supports various options for normalization,
    imputation, and scaling, and allows for reproducibility via a random seed.
    Args:
        calibration_array (np.ndarray): Array of calibrationing features.
        test_array (np.ndarray): Array of test features.
        regularization_array (np.ndarray): Array of regularization features (external reference data).
        calibration_label (np.ndarray): Array of calibrationing labels.
        w_calibration (np.ndarray): Array of calibrationing sample weights.
        n_calibration_in_window (int, optional): Number of calibrationing samples to include in the KNN window. Default is 100.
        frac_regularization_in_window (float, optional): Fraction of regularization samples to include in the KNN window. Default is 0.03.
        normalization (str, optional): Normalization method to use ('rank', etc.). Default is 'rank'.
        impute (bool, optional): Whether to impute missing values. Default is True.
        mi_scaling (bool, optional): Whether to apply mutual information scaling. Default is True.
        seed (int or None, optional): Random seed for reproducibility. Default is None.
    Returns:
        float or np.ndarray: The computed KNN score for the test data using the bootstrapped calibrationing set.
    """
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.choice(calibration_array.shape[0], size=calibration_array.shape[0], replace=True)
    calibration_array_bootstrap = calibration_array[bootstrap_indices]
    calibration_label_bootstrap = calibration_label[bootstrap_indices]
    
    knn_score = get_KNN_score(calibration_array_bootstrap, test_array, regularization_array, calibration_label_bootstrap, w_calibration,  
                              n_calibration_in_window=n_calibration_in_window, frac_regularization_in_window=frac_regularization_in_window, 
                              normalization=normalization, impute=impute, mi_scaling=mi_scaling)

    return knn_score



def get_bootstrap_KNN_score(calibration_array, test_array, regularization_array, 
                            calibration_label, Pprior = None, w_calibration=None, 
                            n_calibration_in_window=100, frac_regularization_in_window=0.03, 
                            normalization='rank', impute=True, mi_scaling=True, n_bootstrap=100, parallel=True):
    """
    Computes bootstrap KNN scores for a given calibration and test set, optionally in parallel.
    This function performs bootstrapped KNN scoring by repeatedly sampling and evaluating the KNN score
    using the provided calibration and test arrays. It supports parallel computation for efficiency.
    Parameters
    ----------
    calibration_array : np.ndarray
        Array of calibration samples (features).
    test_array : np.ndarray
        Array of test samples (features).
    regularization_array : np.ndarray
        Array used for regularization in KNN scoring.
    calibration_label : np.ndarray
        Labels for the calibration samples.
    Pprior : np.ndarray or None, optional
        Prior probabilities for each class (default is None).
    w_calibration : np.ndarray or None, optional
        Weights for calibration samples (default is None).
    n_calibration_in_window : int, optional
        Number of calibration samples to consider in the KNN window (default is 100).
    frac_regularization_in_window : float, optional
        Fraction of regularization samples to include in the window (default is 0.03).
    normalization : str, optional
        Normalization method to use, e.g., 'rank' (default is 'rank').
    impute : bool, optional
        Whether to impute missing values (default is True).
    mi_scaling : bool, optional
        Whether to apply mutual information scaling (default is True).
    n_bootstrap : int, optional
        Number of bootstrap iterations to perform (default is 100).
    parallel : bool, optional
        Whether to run bootstrap iterations in parallel (default is True).
    Returns
    -------
    test_results_array : np.ndarray
        Array of shape (n_test_samples, n_bootstrap) containing the KNN scores for each test sample
        across all bootstrap iterations.
    """
    assert Pprior is not None or w_calibration is not None, "Please provide at least one of Pprior or w_calibration."

    from tqdm import tqdm
    from joblib import Parallel, delayed

    test_results_array = np.zeros((calibration_array.shape[0], n_bootstrap))

    if w_calibration is None:
        w_calibration = get_weight(calibration_label, Pprior)

    if not parallel: 
        for i in tqdm(range(n_bootstrap), desc="Bootstrap Iterations", total=n_bootstrap):
            test_results_array[:, i] = get_one_bootstrap_KNN_score(calibration_array, test_array, regularization_array, calibration_label, w_calibration, 
                                                                   n_calibration_in_window=n_calibration_in_window, frac_regularization_in_window=frac_regularization_in_window,
                                                                   normalization=normalization, impute=impute, mi_scaling=mi_scaling, 
                                                                   seed=i)
    else:
        results = Parallel(n_jobs=-1)(
            delayed(get_one_bootstrap_KNN_score)(calibration_array, test_array, regularization_array, calibration_label, w_calibration, 
                                                 n_calibration_in_window=n_calibration_in_window, frac_regularization_in_window=frac_regularization_in_window,
                                                 normalization=normalization, impute=impute, mi_scaling=mi_scaling, 
                                                 seed=i)
            for i in tqdm(range(n_bootstrap), desc="Parallel Bootstrap Iterations", total=n_bootstrap)
        )

        test_results_array = np.array(results).T # (each test sample, each iteration of bootstrap)

    return test_results_array



def Probability2ACMG_score(Ppost, Pprior, logbase=None):
    if logbase is None:
        logbase = get_logbase(Pprior)

    Likelihood_ratio = Ppost * (1 - Pprior) / ((1 - Ppost) * Pprior)
    ACMG_scores = 8 * np.log(Likelihood_ratio) / np.log(logbase)

    return ACMG_scores



def ACMG_score2Probability(ACMG_scores, Pprior, logbase=None):
    if logbase is None:
        logbase = get_logbase(Pprior)
    
    Pathogenic_likelihood = logbase ** (ACMG_scores / 8)
    Pathogenic_prob = Pathogenic_likelihood * Pprior / (1 + Pprior * (Pathogenic_likelihood-1)) 
    return Pathogenic_prob



def get_P_KNN_ACMG_score(test_results_array, p_value, Pprior, logbase=None):
    """
    Calculate the ACMG scores for pathogenicity and benignity based on P-KNN probabilities.
    This function computes the P-KNN pathogenic and benign probabilities, converts them
    to ACMG scores, and adjusts the scores to be within the range [0, 8]. The final ACMG
    scores are calculated as the difference between the pathogenic and benign scores.
    Parameters:
    -----------
    test_results_array : numpy.ndarray
        A 2D array where each row represents the test results for a sample, and each column
        represents a probability value.
    p_value : float
        The p-value used to determine the index for selecting probabilities. Should be in
        the range (0, 1].
    Pprior : float
        The prior probability of pathogenicity.
    logbase : float, optional
        The logarithmic base used for converting probabilities to ACMG scores. If not
        provided, it will be calculated using the `get_logbase` function.
    Returns:
    --------
    P_KNN_pathogenic : numpy.ndarray
        A 1D array containing the P-KNN pathogenic probabilities for each sample.
    P_KNN_benign : numpy.ndarray
        A 1D array containing the P-KNN benign probabilities for each sample.
    ACMG_scores : numpy.ndarray
        A 1D array containing the ACMG scores for each sample, calculated as the
        difference between the pathogenic and benign scores.
    Notes:
    ------
    - The ACMG scores are clamped to the range [-8, 8].
    - The input `test_results_array` is expected to have at least one column, and the
      `p_value` should be chosen such that the calculated index is valid.
    """

    index = int(np.ceil(p_value*test_results_array.shape[1]))
    P_KNN_pathogenic = np.sort(test_results_array, axis=1)[:, index-1]
    P_KNN_benign = 1- np.sort(test_results_array, axis=1)[:, -index]

    if logbase is None:
        logbase = get_logbase(Pprior) 

    ACMG_pathogenic_scores = Probability2ACMG_score(P_KNN_pathogenic, Pprior, logbase)
    ACMG_benign_scores = Probability2ACMG_score(P_KNN_benign, 1-Pprior, logbase)

    ACMG_pathogenic_scores[ACMG_pathogenic_scores < 0] = 0
    ACMG_benign_scores[ACMG_benign_scores < 0] = 0
    ACMG_pathogenic_scores[ACMG_pathogenic_scores > 8] = 8
    ACMG_benign_scores[ACMG_benign_scores > 8] = 8

    ACMG_scores = ACMG_pathogenic_scores - ACMG_benign_scores

    return P_KNN_pathogenic, P_KNN_benign, ACMG_scores
