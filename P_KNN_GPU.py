import numpy as np
import pandas as pd
import torch
import gc



def get_score_rank_torch(test_tensor: torch.Tensor, calibration_tensor: torch.Tensor):
    """
    Computes the rank scores for each feature in the test tensor based on the sorted values 
    of the corresponding feature in the calibration tensor. Handles NaN values by excluding them 
    during the ranking process.
    Args:
        test_tensor (torch.Tensor): A 2D tensor of shape (n_samples, n_features) containing 
            the test data. May contain NaN values.
        calibration_tensor (torch.Tensor): A 2D tensor of shape (n_samples, n_features) containing 
            the calibrationing data. May contain NaN values.
    Returns:
        torch.Tensor: A 2D tensor of shape (n_samples, n_features) where each element 
            represents the rank score of the corresponding element in the test tensor. 
            NaN values in the test tensor are preserved in the output.
    Notes:
        - The rank score for a value is computed as the average of its left and right 
          positions when inserted into the sorted calibrationing column.
        - NaN values in both test and calibration tensors are ignored during ranking.
        - The output tensor will have the same device as the input test tensor.
    """
    device = calibration_tensor.device
    n_samples, n_features = test_tensor.shape
    rank_scores = []

    for col_idx in range(n_features):
        test_col = test_tensor[:, col_idx]
        calibration_col = calibration_tensor[:, col_idx]

        # exclude NaN, sort calibration_col
        calibration_col_sorted = calibration_col[~torch.isnan(calibration_col)].sort().values

        col_rank = torch.full((n_samples,), float('nan'), device=device)

        non_nan_mask = ~torch.isnan(test_col)
        test_non_nan = test_col[non_nan_mask]

        if test_non_nan.numel() > 0:
            # use searchsorted left and right bounds
            left = torch.searchsorted(calibration_col_sorted, test_non_nan, right=False)
            right = torch.searchsorted(calibration_col_sorted, test_non_nan, right=True)

            rank = (left + right + 1).float() / 2.0
            col_rank[non_nan_mask] = rank

        rank_scores.append(col_rank.unsqueeze(1))

    return torch.cat(rank_scores, dim=1)



def get_rank_torch(calibration_array: np.ndarray, test_array: np.ndarray, regularization_array: np.ndarray):
    """
    Computes the rank scores for calibrationing, testing, and regularization datasets using PyTorch tensors.
    This function converts the input NumPy arrays into PyTorch tensors, moves them to the GPU,
    and calculates rank scores by comparing each dataset against the calibrationing dataset using
    the `get_score_rank_torch` function.
    Args:
        calibration_array (np.ndarray): A NumPy array representing the calibrationing dataset.
        test_array (np.ndarray): A NumPy array representing the testing dataset.
        regularization_array (np.ndarray): A NumPy array representing the regularization dataset.
    Returns:
        tuple: A tuple containing three NumPy arrays:
            - calibration_rank (np.ndarray): Rank scores for the calibrationing dataset.
            - test_rank (np.ndarray): Rank scores for the testing dataset.
            - regularization_rank (np.ndarray): Rank scores for the regularization dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calibration_tensor = torch.tensor(calibration_array, dtype=torch.float64, device=device)
    test_tensor = torch.tensor(test_array, dtype=torch.float64, device=device)
    regularization_tensor = torch.tensor(regularization_array, dtype=torch.float64, device=device)

    calibration_rank = get_score_rank_torch(calibration_tensor, calibration_tensor)
    test_rank = get_score_rank_torch(test_tensor, calibration_tensor)
    regularization_rank = get_score_rank_torch(regularization_tensor, calibration_tensor)

    return calibration_rank.cpu().numpy(), test_rank.cpu().numpy(), regularization_rank.cpu().numpy()



def get_z_torch(calibration_array: np.ndarray, test_array: np.ndarray, regularization_array: np.ndarray):
    """
    Standardizes the input arrays (calibration, test, and regularization) column-wise using PyTorch tensors on GPU.
    For each column in the input arrays:
    - Computes the mean and standard deviation of non-NaN values in the calibrationing array.
    - Uses these statistics to standardize the corresponding column in all three arrays.
    - Handles NaN values by leaving them as NaN in the output.
    Args:
        calibration_array (np.ndarray): 2D NumPy array representing the calibrationing data.
        test_array (np.ndarray): 2D NumPy array representing the test data.
        regularization_array (np.ndarray): 2D NumPy array representing the regularization data.
    Returns:
        tuple: A tuple containing three 2D NumPy arrays:
            - calibration_z (np.ndarray): Standardized calibrationing data.
            - test_z (np.ndarray): Standardized test data.
            - regularization_z (np.ndarray): Standardized regularization data.
    Notes:
        - The function assumes that the input arrays are of the same shape along the second dimension (number of features).
        - NaN values in the input arrays are preserved in the output arrays.
        - The computation is performed on GPU using PyTorch tensors for efficiency.
        - If the standard deviation of a column is zero, it is replaced with 1.0 to avoid division by zero.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calibration_tensor = torch.tensor(calibration_array, dtype=torch.float64, device=device)
    test_tensor = torch.tensor(test_array, dtype=torch.float64, device=device)
    regularization_tensor = torch.tensor(regularization_array, dtype=torch.float64, device=device)

    n_features = calibration_tensor.shape[1]

    # Initialize output tensors
    calibration_z = torch.full_like(calibration_tensor, float('nan'))
    test_z = torch.full_like(test_tensor, float('nan'))
    regularization_z = torch.full_like(regularization_tensor, float('nan'))

    for col in range(n_features):
        calibration_col = calibration_tensor[:, col]
        valid_values = calibration_col[~torch.isnan(calibration_col)]

        if valid_values.numel() > 0:
            mean = torch.mean(valid_values)
            std = torch.std(valid_values)

            std = std if std > 0 else torch.tensor(1.0)  # Avoid division by zero

            calibration_z[:, col] = torch.where(~torch.isnan(calibration_tensor[:, col]), 
                                          (calibration_tensor[:, col] - mean) / std, 
                                          torch.tensor(float('nan')))

            test_z[:, col] = torch.where(~torch.isnan(test_tensor[:, col]), 
                                         (test_tensor[:, col] - mean) / std, 
                                         torch.tensor(float('nan')))

            regularization_z[:, col] = torch.where(~torch.isnan(regularization_tensor[:, col]), 
                                           (regularization_tensor[:, col] - mean) / std, 
                                           torch.tensor(float('nan')))

    return calibration_z.cpu().numpy(), test_z.cpu().numpy(), regularization_z.cpu().numpy()



def silhouette_score_1d_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes the silhouette score for 1-dimensional data using PyTorch tensors, supporting only binary class labels.
    The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
    This implementation is optimized for 1D data and binary labels (0 and 1).
    Args:
        x (torch.Tensor): A 2D tensor of shape (n_samples, 1) containing the data points.
        y (torch.Tensor): A 1D tensor of shape (n_samples,) containing binary labels (0 or 1).
    Returns:
        float: The mean silhouette score for all samples.
    Raises:
        AssertionError: If input shapes are incorrect or labels are not binary.
    Note:
        - Only supports binary labels (0 and 1).
        - Uses mean absolute distance as the distance metric.
        - Excludes self-distance when computing intra-cluster distances.
    """
    assert x.dim() == 2 and x.shape[1] == 1
    assert y.dim() == 1 and x.shape[0] == y.shape[0]
    assert set(y.tolist()) == {0, 1}, "Only binary labels supported"

    y = y.bool()
    x0 = x[~y].view(-1).to(torch.float32)
    x1 = x[y].view(-1).to(torch.float32)

    def mean_abs_distances(xi, xj):
        return torch.abs(xi[:, None] - xj[None, :]).mean(dim=1)

    def mean_abs_distances_exclude_self(xi):
        n = xi.shape[0]
        diff = torch.abs(xi[:, None] - xi[None, :])
        mask = ~torch.eye(n, dtype=torch.bool, device=xi.device)
        return (diff * mask).sum(dim=1) / (n - 1)

    s = torch.zeros_like(x.view(-1), device=x.device, dtype=torch.float32)

    if len(x0) > 1:
        a0 = mean_abs_distances_exclude_self(x0)  # exclude self
    else:
        a0 = torch.zeros_like(x0)
    b0 = mean_abs_distances(x0, x1)
    s0 = (b0 - a0) / torch.clamp(torch.max(a0, b0), min=1e-6)
    s[~y] = s0

    if len(x1) > 1:
        a1 = mean_abs_distances_exclude_self(x1)  # exclude self
    else:
        a1 = torch.zeros_like(x1)
    b1 = mean_abs_distances(x1, x0)
    s1 = (b1 - a1) / torch.clamp(torch.max(a1, b1), min=1e-6)
    s[y] = s1

    return s.mean().item()



class KNNImputerTorch:
    """
    KNNImputerTorch is a PyTorch-based implementation of the k-Nearest Neighbors (k-NN) imputation algorithm 
    for handling missing values in datasets. It computes distances between rows while ignoring missing values 
    and imputes missing entries using the values of the nearest neighbors.
    Attributes:
        n_neighbors (int): Number of neighbors to use for imputation. Default is 5.
        weights (str): Weight function used in prediction. Options are:
            - "uniform": All neighbors are weighted equally.
            - "distance": Closer neighbors are weighted more heavily.
            Default is "uniform".
        missing_values (float): Placeholder for missing values. Default is NaN.
    Methods:
        __init__(n_neighbors=5, weights="uniform", missing_values=float("nan")):
            Initializes the KNNImputerTorch with the specified parameters.
        fit(X):
            Fits the imputer on the input data X by storing a copy of the data and identifying valid columns.
            Args:
                X (torch.Tensor): Input data with potential missing values.
            Returns:
                self: The fitted KNNImputerTorch instance.
        _nan_euclidean_dist(X, Y):
            Computes the pairwise Euclidean distance between rows of X and Y, ignoring missing values.
            Args:
                X (torch.Tensor): First input tensor.
                Y (torch.Tensor): Second input tensor.
            Returns:
                torch.Tensor: Pairwise distances with NaN for rows with no valid comparisons.
        _get_weights(dists):
            Computes weights for neighbors based on the specified weight function.
            Args:
                dists (torch.Tensor): Distances to neighbors.
            Returns:
                torch.Tensor: Weights for each neighbor.
        transform(X):
            Imputes missing values in the input data X using the k-NN algorithm.
            Args:
                X (torch.Tensor): Input data with missing values to be imputed.
            Returns:
                torch.Tensor: Data with missing values imputed.
    """

    def __init__(self, n_neighbors=100, weights="uniform", missing_values=float("nan")):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.missing_values = missing_values
        self._fit_X = None

    def fit(self, X):
        self._fit_X = X.clone()
        self._mask_fit_X = torch.isnan(self._fit_X)
        self._valid_mask = ~torch.all(self._mask_fit_X, dim=0)
        return self

    def _nan_euclidean_dist(self, X, Y):
        # Create masks for NaN values
        X_mask = torch.isnan(X) 
        Y_mask = torch.isnan(Y) 

        # Fill NaN values with 0
        X_filled = torch.where(X_mask, torch.tensor(0.0, device=X.device), X)
        Y_filled = torch.where(Y_mask, torch.tensor(0.0, device=Y.device), Y) 

        # Unsqueeze for broadcasting
        X_filled = X_filled.unsqueeze(1)  # Shape: [n1, 1, n_features]
        Y_filled = Y_filled.unsqueeze(0)  # Shape: [1, n2, n_features]

        # Create valid mask
        valid = ~(X_mask.unsqueeze(1) | Y_mask.unsqueeze(0))  # Shape: [n1, n2, n_features]

        # Compute squared differences
        diff = X_filled - Y_filled  # Shape: [n1, n2, n_features]
        sq_diff = diff ** 2 * valid  # Ignore invalid positions

        # Compute distances
        dist = torch.sqrt(torch.sum(sq_diff, dim=2) / torch.sum(valid, dim=2).clamp(min=1))  # Shape: [5308, 11833]
        dist[torch.sum(valid, dim=2) == 0] = float("nan")  # Handle rows with no valid comparisons

        return dist

    def _get_weights(self, dists):
        if self.weights == "uniform":
            return torch.ones_like(dists, device=dists.device)
        elif self.weights == "distance":
            return 1.0 / torch.clamp(dists, min=1e-5)
        else:
            raise ValueError("Unsupported weight function")

    def transform(self, X, batch_size=512):
        """
        Imputes missing values in the input data X using the k-NN algorithm in batches.
        Args:
            X (torch.Tensor): Input data with missing values to be imputed.
            batch_size (int): Number of rows to process in each batch. Default is 512.
        Returns:
            torch.Tensor: Data with missing values imputed.
        """
        X = X.clone()
        mask = torch.isnan(X)
        row_missing_idx = torch.any(mask[:, self._valid_mask], dim=1).nonzero(as_tuple=True)[0]
    
        # Initialize the output tensor
        X_imputed = X.clone()
    
        # Process in batches
        for start_idx in range(0, row_missing_idx.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, row_missing_idx.shape[0])
            batch_idx = row_missing_idx[start_idx:end_idx]
    
            # Compute distances for the current batch
            dist = self._nan_euclidean_dist(X[batch_idx], self._fit_X)
    
            for col in range(X.shape[1]):
                if not self._valid_mask[col]:
                    continue
    
                receivers_idx = batch_idx[mask[batch_idx, col]]
                if receivers_idx.numel() == 0:
                    continue
    
                donors_idx = (~self._mask_fit_X[:, col]).nonzero(as_tuple=True)[0]
                if donors_idx.numel() == 0:
                    continue
    
                dist_subset = dist[mask[batch_idx, col], :][:, donors_idx]
                all_nan_mask = torch.isnan(dist_subset).all(dim=1)
    
                # Impute with mean if no valid donors
                if all_nan_mask.any():
                    col_mean = torch.nanmean(self._fit_X[:, col])
                    X_imputed[receivers_idx[all_nan_mask], col] = col_mean
                    receivers_idx = receivers_idx[~all_nan_mask]
                    dist_subset = dist_subset[~all_nan_mask]
    
                if receivers_idx.numel() == 0:
                    continue
    
                k = min(self.n_neighbors, donors_idx.shape[0])
                topk = torch.topk(dist_subset, k, largest=False, dim=1)
                indices = topk.indices
                dists = topk.values
    
                weights = self._get_weights(dists)
                values = self._fit_X[donors_idx[indices], col]
                values[torch.isnan(values)] = 0.0
                weights[torch.isnan(values)] = 0.0
    
                weighted_sum = (values * weights).sum(dim=1)
                norm = weights.sum(dim=1).clamp(min=1e-5)
                X_imputed[receivers_idx, col] = weighted_sum / norm
    
        return X_imputed



def KNN_impute_torch(calibration_array: np.ndarray, test_array: np.ndarray, regularization_array: np.ndarray, 
                     n_neighbors: int=100, batch_size: int=512):
    """
    KNN_impute_torch with batch-wise processing for test_array.
    Args:
        calibration_array (numpy.ndarray): Calibrationing dataset.
        test_array (numpy.ndarray): Test dataset to be imputed in batches.
        regularization_array (numpy.ndarray): regularization dataset.
        n_neighbors (int): Number of neighbors to use for imputation.
        batch_size (int): Batch size for processing test_array.
    Returns:
        tuple: Imputed calibration_array, test_array, and regularization_array.
    """
    imputer = KNNImputerTorch(n_neighbors=n_neighbors, weights="uniform")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    calibration_tensor = torch.tensor(calibration_array, dtype=torch.float64, device=device)
    regularization_tensor = torch.tensor(regularization_array, dtype=torch.float64, device=device)
    imputer.fit(calibration_tensor)

    # Impute calibration_array and regularization_array (no batching needed)
    filled_calibration = imputer.transform(calibration_tensor, batch_size).cpu().numpy()
    filled_regularization = imputer.transform(regularization_tensor, batch_size).cpu().numpy()

    # Impute test_array in batches
    # test_tensor = torch.tensor(test_array, dtype=torch.float32)  # Keep on CPU
    test_tensor = torch.tensor(test_array, dtype=torch.float64)  # Keep on CPU
    imputed_test_batches = []

    for start_idx in range(0, test_tensor.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, test_tensor.shape[0])
        test_batch = test_tensor[start_idx:end_idx].to(device)  # Move batch to GPU

        # Apply transform
        imputed_batch = imputer.transform(test_batch, batch_size)

        # Move result back to CPU and append to the results
        imputed_test_batches.append(imputed_batch.cpu())

        # Clear GPU memory
        del test_batch, imputed_batch
        torch.cuda.empty_cache()

    # Concatenate all test batches
    filled_test = torch.cat(imputed_test_batches, dim=0).numpy()

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


    
def get_calibration_percentage_gpu(calibration_array: np.ndarray, test_array: np.ndarray, regularization_array: np.ndarray, 
                                   calibration_label: np.ndarray, w_calibration: float, 
                                   n_calibration_in_window: int = 100, frac_regularization_in_window: float = 0.03, 
                                   batch_size: int = 512):
    """
    Computes the pathogenic percentage for each test sample using calibration and regularization sets on the GPU.
    For each test sample, the function finds a local window defined by the k-nearest calibration and regularization samples,
    then calculates the proportion of pathogenic samples within this window, applying a weighting factor to benign samples.
    Args:
        calibration_array (np.ndarray): Array of calibration samples (shape: [N_calibration, D]).
        test_array (np.ndarray): Array of test samples to evaluate (shape: [N_test, D]).
        regularization_array (np.ndarray): Array of regularization samples (shape: [N_regularization, D]).
        calibration_label (np.ndarray): Binary labels for calibration samples (shape: [N_calibration], 1 for pathogenic, 0 for benign).
        w_calibration (float): Weighting factor applied to benign samples in the denominator.
        n_calibration_in_window (int, optional): Number of nearest calibration samples to include in the window (default: 100).
        frac_regularization_in_window (float, optional): Fraction of regularization samples to include in the window (default: 0.03).
        batch_size (int, optional): Number of test samples to process per GPU batch (default: 512).
    Returns:
        np.ndarray: Array of pathogenic percentages for each test sample (shape: [N_test]).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calibration_tensor = torch.tensor(calibration_array, dtype=torch.float64, device=device)
    regularization_tensor = torch.tensor(regularization_array, dtype=torch.float64, device=device)
    calibration_label_tensor = torch.tensor(calibration_label, dtype=torch.bool, device=device)  # 0/1 label

    # Compute the number of regularization samples in the window
    n_regularization_in_window = int(np.ceil(regularization_tensor.shape[0] * frac_regularization_in_window))

    # Initialize a list to store batch results
    all_percentages = []

    # Process test_array in batches
    for start_idx in range(0, test_array.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, test_array.shape[0])
        test_batch = test_array[start_idx:end_idx]  # Extract batch from test_array

        # Move batch to GPU        
        test_tensor = torch.tensor(test_batch, dtype=torch.float64, device=device)

        # Compute distance matrices
        calibration_distances = torch.cdist(test_tensor, calibration_tensor)       # [B, N_calibration]
        regularization_distances = torch.cdist(test_tensor, regularization_tensor)     # [B, N_regularization]

        # Sort distances
        sorted_calibration_distances, _ = calibration_distances.sort(dim=1)
        sorted_regularization_distances, _ = regularization_distances.sort(dim=1)

        # Determine the distance limit for each test sample
        distance_limit = torch.max(
            sorted_calibration_distances[:, n_calibration_in_window],
            sorted_regularization_distances[:, n_regularization_in_window]
        ).unsqueeze(1)  # [B, 1]

        # Create a mask for calibrationing samples within the distance limit
        mask = calibration_distances <= distance_limit  # [B, N_calibration]

        # Broadcast labels [1, N_calibration] -> [B, N_calibration]
        label_broadcast = calibration_label_tensor.unsqueeze(0)  # [1, N_calibration]

        # Select labels within the distance limit
        selected_labels = mask & label_broadcast  # Logical AND to keep 1's positions

        # Compute counts
        pathogenic_count = selected_labels.sum(dim=1)  # [B]
        total_count = mask.sum(dim=1)                  # [B]
        benign_count = total_count - pathogenic_count  # [B]

        # Compute pathogenic percentages
        denominator = w_calibration * benign_count + pathogenic_count
        pathogenic_percentage = pathogenic_count.float() / (denominator.float() + 1e-10)

        # Move result back to CPU and append to the results
        all_percentages.append(pathogenic_percentage.cpu())

        # Clear GPU memory
        del test_tensor, calibration_distances, regularization_distances, sorted_calibration_distances, sorted_regularization_distances
        torch.cuda.empty_cache()

    # Concatenate all batch results
    return torch.cat(all_percentages).numpy()



def get_KNN_score_gpu(calibration_array: np.ndarray, test_array: np.ndarray, regularization_array: np.ndarray, 
                      calibration_label: np.ndarray, w_calibration: float, 
                      n_calibration_in_window: int = 100, frac_regularization_in_window: float = 0.03, 
                      normalization: str ='rank', impute: bool = True, mi_scaling: bool = True, batch_size: int = 512):
    """
    Computes the K-Nearest Neighbors (KNN) score using GPU acceleration.
    This function preprocesses the input arrays using normalization, imputation, 
    and mutual information scaling (if specified), and then calculates the KNN 
    score based on the calibration percentage.
    Parameters:
        test_array (np.ndarray): The test dataset array.
        calibration_array (np.ndarray): The calibrationing dataset array.
        regularization_array (np.ndarray): The regularization dataset array.
        calibration_label (np.ndarray): The labels corresponding to the calibrationing dataset.
        w_calibration (float): The weight assigned to the calibrationing dataset.
        n_calibration_in_window (int, optional): Number of calibrationing samples in the KNN window. Default is 100.
        frac_regularization_in_window (float, optional): Fraction of regularization samples in the KNN window. Default is 0.03.
        normalization (str, optional): Normalization method to apply. Options are 'rank' or 'z'. Default is 'rank'.
        impute (bool, optional): Whether to perform KNN imputation on missing values. Default is True.
        mi_scaling (bool, optional): Whether to apply mutual information scaling. Default is True.
        batch_size (int, optional): Batch size for GPU processing. Default is 512.
    Returns:
        float: The computed KNN score.
    Notes:
        - Please manually exclude rows with all NaN values in the input arrays before calling this function.
        - The function assumes that the input arrays are compatible with GPU processing.
        - The `get_rank_torch`, `get_z_torch`, `KNN_impute_gpu`, `mutual_information_scaling`, 
            and `get_calibration_percentage_gpu` functions must be implemented and available 
            in the same context for this function to work.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if normalization == 'rank':
        calibration_array, test_array, regularization_array = get_rank_torch(calibration_array, test_array, regularization_array)
    elif normalization == 'z':
        calibration_array, test_array, regularization_array = get_z_torch(calibration_array, test_array, regularization_array)

    if not impute:
        valid_calibration_idx = np.all(~np.isnan(calibration_array), axis=1)
        calibration_array = calibration_array[valid_calibration_idx]
        calibration_label = calibration_label[valid_calibration_idx]

    calibration_array, test_array, regularization_array = KNN_impute_torch(calibration_array, test_array, regularization_array, 
                                                             n_neighbors = n_calibration_in_window, batch_size = batch_size)
    
    if mi_scaling:
        calibration_array, test_array, regularization_array, _ = mutual_information_scaling(calibration_array, test_array, regularization_array, calibration_label)

    knn_score = get_calibration_percentage_gpu(calibration_array, test_array, regularization_array, 
                                               calibration_label, w_calibration, 
                                               n_calibration_in_window, frac_regularization_in_window, 
                                               batch_size)
    return knn_score



def get_bootstrap_KNN_score_gpu(calibration_array: np.ndarray, test_array: np.ndarray, regularization_array: np.ndarray, 
                                calibration_label: np.ndarray, Pprior: float=None, w_calibration: float=None, 
                                n_calibration_in_window: int = 100, frac_regularization_in_window: float = 0.03, 
                                normalization: str ='rank', impute: bool = True, mi_scaling: bool = True, n_bootstrap: int = 100, batch_size: int = 512):
    """
    Computes bootstrap scores for a K-Nearest Neighbors (KNN) model using the provided calibrationing and test data.
    This version is optimized for a single GPU (no parallel processing). 
    Parameters:
        calibration_array (numpy.ndarray): The calibrationing data array.
        test_array (numpy.ndarray): The test data array.
        regularization_array (numpy.ndarray): The regularization data array used for additional reference.
        calibration_label (numpy.ndarray): Labels corresponding to the calibrationing data.
        Pprior (float, optional): Prior probability of being benign. If None, it is calculated from the calibration_label.
        w_calibration (float, optional): Weight for benign labels. If None, it is calculated using `get_weight`.
        normalization (str, optional): The normalization method to apply. Default is 'rank'.
        impute (bool, optional): Whether to perform imputation on missing data. Default is True.
        mi_scaling (bool, optional): Whether to apply mutual information scaling. Default is True.
        n_calibration_in_window (int, optional): Number of calibrationing samples to consider in the KNN window. Default is 100.
        frac_regularization_in_window (float, optional): Fraction of regularization samples to include in the KNN window. Default is 0.03.
        n_bootstrap (int, optional): Number of bootstrap iterations to perform. Default is 100.
    Returns:
        numpy.ndarray: A 2D array of shape (number of test samples, n_bootstrap), where each element represents 
                       the bootstrap score for a test sample in a specific iteration.
    Notes:
        - Input at least one of Pprior or w_calibration.
    """
    assert Pprior is not None or w_calibration is not None, "Please provide at least one of Pprior or w_calibration."

    from tqdm import tqdm
    test_results_array = np.zeros((test_array.shape[0], n_bootstrap))

    if w_calibration is None:
        w_calibration = get_weight(calibration_label, Pprior)

    for i in tqdm(range(n_bootstrap), desc="Bootstrap Iterations", total=n_bootstrap):
        rng = np.random.default_rng(i)
        bootstrap_indices = rng.choice(calibration_array.shape[0], size=calibration_array.shape[0], replace=True)
        calibration_array_bootstrap = calibration_array[bootstrap_indices]
        calibration_label_bootstrap = calibration_label[bootstrap_indices]

        # Perform one bootstrap iteration

        test_results_array[:, i] = get_KNN_score_gpu(calibration_array_bootstrap, test_array, regularization_array, 
                                                     calibration_label_bootstrap, w_calibration, 
                                                     n_calibration_in_window, frac_regularization_in_window, 
                                                     normalization, impute, mi_scaling, batch_size)
        torch.cuda.empty_cache()
        gc.collect()

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



def get_P_KNN_ACMG_score_1D(test_results_array, test_score, p_value, Pprior, logbase=None):
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

    tool_score = test_score.flatten()

    # Deal with pathogenic
    sorted_indices = np.argsort(-tool_score) # from largest to smallest
    sorted_pathogenic_prob = test_results_array[sorted_indices]

    sorted_pathogenic_prob = np.minimum.accumulate(sorted_pathogenic_prob, axis=0)
    
    index = int(np.ceil(p_value*test_results_array.shape[1]))
    CI_pathogenic_prob = np.sort(sorted_pathogenic_prob, axis=1)[:, index-1]

    P_KNN_pathogenic = np.zeros_like(CI_pathogenic_prob)
    P_KNN_pathogenic[sorted_indices] = CI_pathogenic_prob

    # Deal with benign
    sorted_indices = np.argsort(tool_score) # from smallest to largest
    sorted_benign_prob = test_results_array[sorted_indices]
    sorted_benign_prob = 1- sorted_benign_prob

    sorted_benign_prob = np.minimum.accumulate(sorted_benign_prob, axis=0)
    
    CI_benign_prob = np.sort(sorted_benign_prob, axis=1)[:, index-1]

    P_KNN_benign = np.zeros_like(CI_benign_prob)
    P_KNN_benign[sorted_indices] = CI_benign_prob

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



def weighted_score_with_binom_ci(p_array, n_array, w, p_value=0.05):
    from scipy.stats import binom
    p_array = np.asarray(p_array)
    n_array = np.asarray(n_array)
    shape = p_array.shape

    scores = np.full(shape, np.nan)
    ci_lower = np.full(shape, np.nan)
    ci_upper = np.full(shape, np.nan)

    t_array = p_array + n_array

    it = np.nditer(p_array, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        p = p_array[idx]
        t = t_array[idx]

        if t > 0:
            pi_hat = p / t
            ci_low, ci_up = binom.interval(1-p_value, t, pi_hat)
            
            def weighted(pi):
                return pi / (pi + w * (1 - pi))

            scores[idx] = weighted(pi_hat)
            ci_lower[idx] = weighted(ci_low / t)
            ci_upper[idx] = weighted(ci_up / t)
        
        it.iternext()
    
    return scores, ci_lower, ci_upper



def evaluate_result_1D(test_results_array, test_score, test_label, p_value, Pprior, logbase=None, category = None, show_plot=True, save_name=None):
    if logbase is None:
        logbase = get_logbase(Pprior)
    
    w_test = (1 - Pprior) * sum(test_label == 1) / (sum(test_label == 0) * Pprior)  
    
    # Calculate the pathogenic and benign threshold for evidence level
    Post_p = np.zeros(4) 
    Post_b = np.zeros(4)

    for j in range(4):
        Post_p[j] = logbase ** (1 / 2 ** j) * Pprior / ((logbase ** (1 / 2 ** j) - 1) * Pprior + 1)
        Post_b[j] = (logbase ** (1 / 2 ** j)) * (1 - Pprior) / (((logbase ** (1 / 2 ** j)) - 1) * (1 - Pprior) + 1)

    tool_score = test_score.flatten()

    # Deal with pathogenic
    sorted_indices = np.argsort(-tool_score) # from largest to smallest
    sorted_pathogenic_prob = test_results_array[sorted_indices]

    sorted_pathogenic_prob = np.minimum.accumulate(sorted_pathogenic_prob, axis=0)
    
    index = int(np.ceil(p_value*test_results_array.shape[1]))
    CI_pathogenic_prob = np.sort(sorted_pathogenic_prob, axis=1)[:, index-1]

    P_KNN_pathogenic = np.zeros_like(CI_pathogenic_prob)
    P_KNN_pathogenic[sorted_indices] = CI_pathogenic_prob

    # Deal with benign
    sorted_indices = np.argsort(tool_score) # from smallest to largest
    sorted_benign_prob = test_results_array[sorted_indices]
    sorted_benign_prob = 1- sorted_benign_prob

    sorted_benign_prob = np.minimum.accumulate(sorted_benign_prob, axis=0)
    
    CI_benign_prob = np.sort(sorted_benign_prob, axis=1)[:, index-1]

    P_KNN_benign = np.zeros_like(CI_benign_prob)
    P_KNN_benign[sorted_indices] = CI_benign_prob

    # Calculate ACMG scores
    ACMG_pathogenic_scores = Probability2ACMG_score(P_KNN_pathogenic, Pprior, logbase)
    ACMG_benign_scores = Probability2ACMG_score(P_KNN_benign, 1-Pprior, logbase)

    ACMG_pathogenic_scores[ACMG_pathogenic_scores < 0] = 0
    ACMG_benign_scores[ACMG_benign_scores < 0] = 0
    ACMG_pathogenic_scores[ACMG_pathogenic_scores > 8] = 8
    ACMG_benign_scores[ACMG_benign_scores > 8] = 8

    ACMG_scores = ACMG_pathogenic_scores - ACMG_benign_scores

    # Divide the results into pathogenic and benign based on test_label
    Pathogenic_P_KNN_pathogenic = P_KNN_pathogenic[test_label == 1]
    Benign_P_KNN_pathogenic = P_KNN_pathogenic[test_label == 0]
    Pathogenic_P_KNN_benign = P_KNN_benign[test_label == 1]
    Benign_P_KNN_benign = P_KNN_benign[test_label == 0]
    Pathogenic_ACMG_scores = ACMG_scores[test_label == 1]
    Benign_ACMG_scores = ACMG_scores[test_label == 0]
    
    # Print correct and incorrect assignment of pathogenic and benign variants
    print("Pathogenic evidence")
    accumulate_pathogenic_count = 0
    accumulate_benign_count = 0
    for ACMGevidence, threshold in zip(["+8", "+4", "+2", "+1"], Post_p):
        pathogenic_count = (Pathogenic_P_KNN_pathogenic > threshold).sum() - accumulate_pathogenic_count
        benign_count = (Benign_P_KNN_pathogenic > threshold).sum() - accumulate_benign_count
        accumulate_pathogenic_count += pathogenic_count
        accumulate_benign_count += benign_count
        print(f"  Evidence score {ACMGevidence} Probability threshold {threshold:.3f}:")
        print(f"    Pathogenic {pathogenic_count} ({pathogenic_count/(len(Pathogenic_P_KNN_pathogenic)):.2%}) pathogenic variants")
        print(f"    Benign {benign_count} ({benign_count/(len(Benign_P_KNN_pathogenic)):.2%}) error benign variants")
        print(f"    Weighted correct rate: {pathogenic_count/(pathogenic_count+benign_count*w_test):.2%}")

    print("Benign evidence")
    accumulate_pathogenic_count = 0
    accumulate_benign_count = 0
    for ACMGevidence, threshold in zip(["+8", "+4", "+2", "+1"], Post_b):
        pathogenic_count = (Pathogenic_P_KNN_benign > threshold).sum() - accumulate_pathogenic_count
        benign_count = (Benign_P_KNN_benign > threshold).sum() - accumulate_benign_count
        accumulate_pathogenic_count += pathogenic_count
        accumulate_benign_count += benign_count
        print(f"  Evidence score {ACMGevidence} Probability threshold {threshold:.3f}:")
        print(f"    Benign {benign_count} ({benign_count/(len(Benign_P_KNN_benign)):.2%}) benign variants")
        print(f"    Pathogenic {pathogenic_count} ({pathogenic_count/(len(Pathogenic_P_KNN_benign)):.2%}) error pathogenic variants")
        print(f"    Weighted correct rate: {benign_count*w_test/(pathogenic_count+benign_count*w_test):.2%}")

    # Evidence violin plot
    evidence_strength_data = pd.DataFrame({
        "Score": np.concatenate([-Benign_ACMG_scores, Pathogenic_ACMG_scores]),
        "Label": ["Benign variants"] * len(Benign_ACMG_scores) + ["Pathogenic variants"] * len(Pathogenic_ACMG_scores),
        "Category": [category] * (len(Benign_ACMG_scores) + len(Pathogenic_ACMG_scores))
        })

    if show_plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="ticks")
    
        plt.figure(figsize=(6, 6))
        sns.violinplot(
            x="Category",
            y="Score", 
            hue="Label",   # Pathogenic and Benign 
            data=evidence_strength_data, 
            split=True,   # on each side of violin
            inner="box", 
            palette={"Pathogenic variants": "red", "Benign variants": "blue"},
            alpha=0.6, 
            density_norm='area'
        )
        
        plt.xlabel("")
        plt.ylabel("Evidence strength (LLR)", fontsize=14)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=14)
        sns.despine(top=True, right=True)
        
        if save_name:
            plt.savefig(f"{save_name}_ACMGevidence.svg", format="svg", bbox_inches='tight')
        plt.show()

    # Pathogenic Calibration Plot with Confidence Interval
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    
    pathogenic_counts, _ = np.histogram(Pathogenic_P_KNN_pathogenic, bins=bins)
    benign_counts, _ = np.histogram(Benign_P_KNN_pathogenic, bins=bins)

    pathogenic_ratios, ci_lower, ci_upper = weighted_score_with_binom_ci(pathogenic_counts, benign_counts, w_test, p_value)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid_mask = ~np.isnan(pathogenic_ratios) & ~np.isnan(ci_lower) & ~np.isnan(ci_upper)

    pathogenic_calibration_dict = {}

    pathogenic_calibration_dict['bin_centers'] = bin_centers[valid_mask]
    pathogenic_calibration_dict['ratios'] = pathogenic_ratios[valid_mask]
    pathogenic_calibration_dict['ci_lower'] = ci_lower[valid_mask]
    pathogenic_calibration_dict['ci_upper'] = ci_upper[valid_mask]


    bins = np.linspace(0.95, 1, num_bins + 1)

    pathogenic_counts, _ = np.histogram(Pathogenic_P_KNN_benign, bins=bins)
    benign_counts, _ = np.histogram(Benign_P_KNN_benign, bins=bins)
    
    benign_ratios, ci_lower, ci_upper = weighted_score_with_binom_ci(benign_counts, pathogenic_counts, 1/w_test, p_value)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid_mask = ~np.isnan(benign_ratios) & ~np.isnan(ci_lower) & ~np.isnan(ci_upper)

    benign_calibration_dict = {}
    benign_calibration_dict['bin_centers'] = bin_centers[valid_mask]
    benign_calibration_dict['ratios'] = benign_ratios[valid_mask]
    benign_calibration_dict['ci_lower'] = ci_lower[valid_mask]
    benign_calibration_dict['ci_upper'] = ci_upper[valid_mask]

    if show_plot:
        # Plot the calibration curve for pathogenic
        plt.figure(figsize=(6, 6))
        plt.plot(pathogenic_calibration_dict['bin_centers'], pathogenic_calibration_dict['ratios'], 
                 'o-', color='red', label='Frequency of pathogenic variants')
        plt.fill_between(pathogenic_calibration_dict['bin_centers'], 
                         pathogenic_calibration_dict['ci_lower'], pathogenic_calibration_dict['ci_upper'], 
                         color='red', alpha=0.2, label='95% Confidence Interval')
        plt.xlabel("Posterior probability (pathogenic)", fontsize=14)
        plt.ylabel("Frequency of pathogenic variants", fontsize=14)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=14)
        
        x_vals = np.linspace(0, 1, 100)
        plt.plot(x_vals, x_vals, label="y = x", color='red', linestyle='--', alpha=0.2)
        
        for threshold in Post_p:
            plt.axvline(x=threshold, color='orange', linestyle='--', label=f'X = {threshold}', alpha=0.2)
            plt.axhline(y=threshold, color='orange', linestyle='--', label=f'Y = {threshold}', alpha=0.2)

        sns.despine(top=True, right=True)   
        if save_name:
            plt.savefig(f"{save_name}_P_Calibration.svg", format="svg", bbox_inches='tight')

        plt.show()

        # Plot the calibration curve for benign
        plt.figure(figsize=(6, 6))
        plt.plot(benign_calibration_dict['bin_centers'], benign_calibration_dict['ratios'],
                 'o-', color='blue', label='Frequency of benign variants')
        plt.fill_between(benign_calibration_dict['bin_centers'],
                         benign_calibration_dict['ci_lower'], benign_calibration_dict['ci_upper'],
                        color='blue', alpha=0.2, label='95% Confidence Interval')
        plt.xlabel("Posterior probability (benign)", fontsize=14)
        plt.ylabel("Frequency of benign variants", fontsize=14)
        plt.ylim(0.95, 1)
        plt.xlim(0.95, 1)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=14)

        x_vals = np.linspace(0.95, 1, 100)
        plt.plot(x_vals, x_vals, label="y = x", color='blue', linestyle='--', alpha=0.2)
        
        for threshold in Post_b:
            plt.axvline(x=threshold, color='cyan', linestyle='--', label=f'X = {threshold}', alpha=0.2)
            plt.axhline(y=threshold, color='cyan', linestyle='--', label=f'Y = {threshold}', alpha=0.2)

        sns.despine(top=True, right=True)
        if save_name:
            plt.savefig(f"{save_name}_B_Calibration.svg", format="svg", bbox_inches='tight')

        plt.show()

    return evidence_strength_data, pathogenic_calibration_dict, benign_calibration_dict



def evaluate_result(test_results_array, test_label, p_value, Pprior, logbase=None, category = None, show_plot=True, save_name=None):
    if logbase is None:
        logbase = get_logbase(Pprior)
    
    w_test = (1 - Pprior) * sum(test_label == 1) / (sum(test_label == 0) * Pprior)  
    
    # Calculate the pathogenic and benign threshold for evidence level
    Post_p = np.zeros(4) 
    Post_b = np.zeros(4)

    for j in range(4):
        Post_p[j] = logbase ** (1 / 2 ** j) * Pprior / ((logbase ** (1 / 2 ** j) - 1) * Pprior + 1)
        Post_b[j] = (logbase ** (1 / 2 ** j)) * (1 - Pprior) / (((logbase ** (1 / 2 ** j)) - 1) * (1 - Pprior) + 1)

    Pathogenic_test_results_array = test_results_array[test_label == 1]
    Benign_test_results_array = test_results_array[test_label == 0]

    Pathogenic_P_KNN_pathogenic, Pathogenic_P_KNN_benign, Pathogenic_ACMG_scores = get_P_KNN_ACMG_score(
        Pathogenic_test_results_array, p_value, Pprior, logbase)
    
    Benign_P_KNN_pathogenic, Benign_P_KNN_benign, Benign_ACMG_scores = get_P_KNN_ACMG_score(
        Benign_test_results_array, p_value, Pprior, logbase) 
    
    # Print correct and incorrect assignment of pathogenic and benign variants
    print("Pathogenic evidence")
    accumulate_pathogenic_count = 0
    accumulate_benign_count = 0
    for ACMGevidence, threshold in zip(["+8", "+4", "+2", "+1"], Post_p):
        pathogenic_count = (Pathogenic_P_KNN_pathogenic > threshold).sum() - accumulate_pathogenic_count
        benign_count = (Benign_P_KNN_pathogenic > threshold).sum() - accumulate_benign_count
        accumulate_pathogenic_count += pathogenic_count
        accumulate_benign_count += benign_count
        print(f"  Evidence score {ACMGevidence} Probability threshold {threshold:.3f}:")
        print(f"    Pathogenic {pathogenic_count} ({pathogenic_count/(len(Pathogenic_P_KNN_pathogenic)):.2%}) pathogenic variants")
        print(f"    Benign {benign_count} ({benign_count/(len(Benign_P_KNN_pathogenic)):.2%}) error benign variants")
        print(f"    Weighted correct rate: {pathogenic_count/(pathogenic_count+benign_count*w_test):.2%}")

    print("Benign evidence")
    accumulate_pathogenic_count = 0
    accumulate_benign_count = 0
    for ACMGevidence, threshold in zip(["+8", "+4", "+2", "+1"], Post_b):
        pathogenic_count = (Pathogenic_P_KNN_benign > threshold).sum() - accumulate_pathogenic_count
        benign_count = (Benign_P_KNN_benign > threshold).sum() - accumulate_benign_count
        accumulate_pathogenic_count += pathogenic_count
        accumulate_benign_count += benign_count
        print(f"  Evidence score {ACMGevidence} Probability threshold {threshold:.3f}:")
        print(f"    Benign {benign_count} ({benign_count/(len(Benign_P_KNN_benign)):.2%}) benign variants")
        print(f"    Pathogenic {pathogenic_count} ({pathogenic_count/(len(Pathogenic_P_KNN_benign)):.2%}) error pathogenic variants")
        print(f"    Weighted correct rate: {benign_count*w_test/(pathogenic_count+benign_count*w_test):.2%}")

    # Evidence violin plot
    evidence_strength_data = pd.DataFrame({
        "Score": np.concatenate([-Benign_ACMG_scores, Pathogenic_ACMG_scores]),
        "Label": ["Benign variants"] * len(Benign_ACMG_scores) + ["Pathogenic variants"] * len(Pathogenic_ACMG_scores),
        "Category": [category] * (len(Benign_ACMG_scores) + len(Pathogenic_ACMG_scores))
        })

    if show_plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="ticks")
    
        plt.figure(figsize=(6, 6))
        sns.violinplot(
            x="Category",
            y="Score", 
            hue="Label",   # Pathogenic and Benign 
            data=evidence_strength_data, 
            split=True,   # on each side of violin
            inner="box", 
            palette={"Pathogenic variants": "red", "Benign variants": "blue"},
            alpha=0.6, 
            density_norm='area'
        )
        
        plt.xlabel("")
        plt.ylabel("Evidence strength (LLR)", fontsize=14)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=14)
        sns.despine(top=True, right=True)
        
        if save_name:
            plt.savefig(f"{save_name}_ACMGevidence.svg", format="svg", bbox_inches='tight')
        plt.show()

    # Pathogenic Calibration Plot with Confidence Interval
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    
    pathogenic_counts, _ = np.histogram(Pathogenic_P_KNN_pathogenic, bins=bins)
    benign_counts, _ = np.histogram(Benign_P_KNN_pathogenic, bins=bins)

    pathogenic_ratios, ci_lower, ci_upper = weighted_score_with_binom_ci(pathogenic_counts, benign_counts, w_test, p_value)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid_mask = ~np.isnan(pathogenic_ratios) & ~np.isnan(ci_lower) & ~np.isnan(ci_upper)

    pathogenic_calibration_dict = {}

    pathogenic_calibration_dict['bin_centers'] = bin_centers[valid_mask]
    pathogenic_calibration_dict['ratios'] = pathogenic_ratios[valid_mask]
    pathogenic_calibration_dict['ci_lower'] = ci_lower[valid_mask]
    pathogenic_calibration_dict['ci_upper'] = ci_upper[valid_mask]


    bins = np.linspace(0.95, 1, num_bins + 1)

    pathogenic_counts, _ = np.histogram(Pathogenic_P_KNN_benign, bins=bins)
    benign_counts, _ = np.histogram(Benign_P_KNN_benign, bins=bins)
    
    benign_ratios, ci_lower, ci_upper = weighted_score_with_binom_ci(benign_counts, pathogenic_counts, 1/w_test, p_value)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid_mask = ~np.isnan(benign_ratios) & ~np.isnan(ci_lower) & ~np.isnan(ci_upper)

    benign_calibration_dict = {}
    benign_calibration_dict['bin_centers'] = bin_centers[valid_mask]
    benign_calibration_dict['ratios'] = benign_ratios[valid_mask]
    benign_calibration_dict['ci_lower'] = ci_lower[valid_mask]
    benign_calibration_dict['ci_upper'] = ci_upper[valid_mask]

    if show_plot:
        # Plot the calibration curve for pathogenic
        plt.figure(figsize=(6, 6))
        plt.plot(pathogenic_calibration_dict['bin_centers'], pathogenic_calibration_dict['ratios'], 
                 'o-', color='red', label='Frequency of pathogenic variants')
        plt.fill_between(pathogenic_calibration_dict['bin_centers'], 
                         pathogenic_calibration_dict['ci_lower'], pathogenic_calibration_dict['ci_upper'], 
                         color='red', alpha=0.2, label='95% Confidence Interval')
        plt.xlabel("P-KNN posterior probability (pathogenic)", fontsize=14)
        plt.ylabel("Frequency of pathogenic variants", fontsize=14)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=14)

        x_vals = np.linspace(0, 1, 100)
        plt.plot(x_vals, x_vals, label="y = x", color='red', linestyle='--', alpha=0.2)
        
        for threshold in Post_p:
            plt.axvline(x=threshold, color='orange', linestyle='--', label=f'X = {threshold}', alpha=0.2)
            plt.axhline(y=threshold, color='orange', linestyle='--', label=f'Y = {threshold}', alpha=0.2)

        sns.despine(top=True, right=True)   
        if save_name:
            plt.savefig(f"{save_name}_P_Calibration.svg", format="svg", bbox_inches='tight')        

        plt.show()

        # Plot the calibration curve for benign
        plt.figure(figsize=(6, 6))
        plt.plot(benign_calibration_dict['bin_centers'], benign_calibration_dict['ratios'],
                 'o-', color='blue', label='Frequency of benign variants')
        plt.fill_between(benign_calibration_dict['bin_centers'],
                         benign_calibration_dict['ci_lower'], benign_calibration_dict['ci_upper'],
                        color='blue', alpha=0.2, label='95% Confidence Interval')
        plt.xlabel("P-KNN posterior probability (benign)", fontsize=14)
        plt.ylabel("Frequency of benign variants", fontsize=14)
        plt.ylim(0.95, 1)
        plt.xlim(0.95, 1)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=14)
        
        x_vals = np.linspace(0.95, 1, 100)
        plt.plot(x_vals, x_vals, label="y = x", color='blue', linestyle='--', alpha=0.2)
        
        for threshold in Post_b:
            plt.axvline(x=threshold, color='cyan', linestyle='--', label=f'X = {threshold}', alpha=0.2)
            plt.axhline(y=threshold, color='cyan', linestyle='--', label=f'Y = {threshold}', alpha=0.2)

        sns.despine(top=True, right=True)
        if save_name:
            plt.savefig(f"{save_name}_B_Calibration.svg", format="svg", bbox_inches='tight')  
            
        plt.show()

    return evidence_strength_data, pathogenic_calibration_dict, benign_calibration_dict