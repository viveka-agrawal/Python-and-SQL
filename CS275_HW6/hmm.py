
# UCI CS275P: Hidden Markov Modeling of Character Sequences

import numpy as np
import matplotlib.pyplot as plt

# HMM support package:
# pip install hmmlearn
# or 
# conda install -c conda-forge hmmlearn
from hmmlearn import hmm

from text_utils import text_to_num, num_to_text, read_text_file, sample_hmm_text, erase_random_chars

def evaluate_hmm_model(model, data):
    """
    Compute the log-likelihood, AIC, and BIC for a trained Hidden Markov Model.

    Parameters:
    ----------
    model : hmmlearn.hmm.CategoricalHMM
        A trained categorical HMM model.
    data : np.ndarray of shape (N,)
        1D array of integers representing encoded character observations.

    Returns:
    -------
    log_likelihood : float
        Log-likelihood of the data under the model.
    aic : float
        Akaike Information Criterion score.
    bic : float
        Bayesian Information Criterion score.
    """
    # calculate log-likelihood of data
    log_likelihood = model.score(np.reshape(data, (-1, 1)))
    
    # calculate number of params (dof)
    # M states & W = 27 obs:
    #    initial state probs: M - 1 free params bc of normalization
    #    transition probs: M * (M - 1) free params 
    #    emission probs: M * (W - 1) free params
    M = model.n_components
    W = 27  # valid chars numbers (a-z & underscore)
    dof = (M - 1) + M * (M - 1) + M * (W - 1)
    
    # calculate AIC & BIC
    num_obs = len(data)  
    aic = log_likelihood - dof
    bic = log_likelihood - (0.5 * dof * np.log(num_obs))

    return log_likelihood, aic, bic


def denoise_text(model, noisy_data, noisy_text, erase_rate=0.2, num_valid_characters=27):
    """
    Infer the most likely hidden states and restore erased characters using a trained HMM.

    Parameters:
    ----------
    model : hmmlearn.hmm.CategoricalHMM
        Trained HMM on clean data.
    noisy_data : np.ndarray of shape (N,)
        1D array of integers including erased character codes (0 to num_valid_characters).
    noisy_text : np.ndarray of shape (N,)
        Original text array with erased characters as '*'.
    erase_rate : float, optional
        Probability of character erasure, used to modify emission matrix. Default is 0.2.
    num_valid_characters : int, optional
        Total number of valid character symbols (typically 27 for a-z and '_'). Default is 27.

    Returns:
    -------
    noisy_state_prob : np.ndarray of shape (N, M)
        Posterior state probabilities for each position, where M is number of HMM states.
    denoised_data : np.ndarray of shape (N,)
        The reconstructed data with erased values filled in.
    """
    M = model.n_components

    # Modify emission matrix to account for erasures
    erase_prob = np.full((num_valid_characters, num_valid_characters + 1), erase_rate / num_valid_characters)
    np.fill_diagonal(erase_prob[:, :num_valid_characters], 1 - erase_rate)

    # calculate modified emission probs
    obs_mat = np.matmul(model.emissionprob_, erase_prob)

    noisy_model = hmm.CategoricalHMM(n_components=M)
    noisy_model.n_features = num_valid_characters + 1
    noisy_model.startprob_ = model.startprob_
    noisy_model.transmat_ = model.transmat_
    noisy_model.emissionprob_ = obs_mat

    # calculate posterior state probs
    noisy_state_prob = noisy_model.predict_proba(noisy_data.reshape(-1, 1))

    # calculate p(x_t | y) = sum_z p(x_t | z) * p(z | y)
    p_xt_given_y = np.dot(noisy_state_prob, model.emissionprob_)

    # replace erased entries w/ highly likely characters 
    denoised_data = np.copy(noisy_data)
    erase_mask = (noisy_data == num_valid_characters)
    denoised_data[erase_mask] = np.argmax(p_xt_given_y[erase_mask], axis=1)

    return noisy_state_prob, denoised_data


if __name__ == "__main__":
    # If we should load from save files or train the models
    load_from_saves = False 

    # Load the data 
    train_text = read_text_file("aliceTrain.txt")
    test_text = read_text_file("aliceTest.txt")

    # Convert from ASCII characters to numbers
    train_data = text_to_num(train_text)
    test_data = text_to_num(test_text)

    # Constants
    num_valid_characters = 27

    # Parameters
    max_num_em_steps = 500
    em_change_tolerance = 1e-6

    # The different latent space sizes (num of states) we want to try
    values_of_M_to_try = [1, 5, 10, 15, 20, 30, 40, 50, 60]

    models = []
    training_log_likelihoods = []
    if(load_from_saves is False):
    ## Train the models
        for M in values_of_M_to_try:

            print("")
            print("*** Training with {:d} states: ***".format(M))

            # Create the HMM
            model = hmm.CategoricalHMM(n_components=M, verbose=True, \
                n_iter=max_num_em_steps, tol=em_change_tolerance)

            # Fit the HMM to the data
            model.fit(np.reshape(train_data, (-1, 1)))

            # Save data
            models.append(model)
            training_log_likelihoods.append(list(model.monitor_.history))

            # Checkpoint the model in case we crash...
            save_dict = dict()
            save_dict["n_features"] = model.n_features
            save_dict["emissionprob_"] = model.emissionprob_
            save_dict["transmat_"] = model.transmat_
            save_dict["startprob_"] = model.startprob_
            save_dict["training_log_likelihoods"] = np.asarray(list(model.monitor_.history))
            np.save("model_m{:03}.npy".format(M), save_dict)

    else:
        ## Load the models from a save file (so that we dont have to do all the 
        ## expensive training if we already trained)
        for M in values_of_M_to_try:

            # Load the data
            data = np.load("model_m{:03}.npy".format(M), allow_pickle=True).item()

            model = hmm.CategoricalHMM(n_components=M, verbose=True)
            model.n_features = data["n_features"]
            model.emissionprob_ = data["emissionprob_"]
            model.transmat_ = data["transmat_"]
            model.startprob_ = data["startprob_"]

            tll = data["training_log_likelihoods"]

            models.append(model)
            training_log_likelihoods.append(tll.tolist())

    ## Compute AIC/BIC scores & test likelihoods for all models 
    train_AIC = np.zeros((len(values_of_M_to_try),))
    train_BIC = np.zeros((len(values_of_M_to_try),))
    test_log_likelihoods = np.zeros((len(values_of_M_to_try),))
    max_train_log_likelihoods = np.zeros((len(values_of_M_to_try),))

    for m_idx in range(len(values_of_M_to_try)):
        max_train_log_likelihoods[m_idx], train_AIC[m_idx], train_BIC[m_idx] = evaluate_hmm_model(models[m_idx], train_data)
        test_log_likelihoods[m_idx], _, _ = evaluate_hmm_model(models[m_idx], test_data)

    # Plot them
    fig1 = plt.figure(1)
    plt.plot(values_of_M_to_try, max_train_log_likelihoods, label="Training Log-Like")
    plt.plot(values_of_M_to_try, train_AIC, label="AIC Criterion")
    plt.plot(values_of_M_to_try, train_BIC, label="BIC Criterion")
    plt.plot(values_of_M_to_try, test_log_likelihoods, label="Test Log-Like")
    plt.xlabel("Number of States")
    plt.ylabel("Log-likelihood")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # QUESTION 2c
    # index of model w/ highest AIC
    best_aic_i = np.argmax(train_AIC)  
    # index of model w/ highest BIC
    best_bic_i = np.argmax(train_BIC)  
    # model w/ best test log-likelihood
    best_test_i = np.argmax(test_log_likelihoods)  

    print(f"\n Results (2c): ")
    print(f"Best AIC: M = {values_of_M_to_try[best_aic_i]} (AIC = {train_AIC[best_aic_i]:.3f})")
    print(f"Best BIC: M = {values_of_M_to_try[best_bic_i]} (BIC = {train_BIC[best_bic_i]:.3f})")
    print(f"Best Test: M = {values_of_M_to_try[best_test_i]} (Test LogLikelihood = {test_log_likelihoods[best_test_i]:.3f})")

    # QUESTION 2d
    print("\n Results (2d): ")
    # models to sample from: M = 1, best AIC, best BIC, M = 60
    models_for_sampling = [
        (0, "M = 1 (No sequential dependence)"),
        (best_aic_i, f"M = {values_of_M_to_try[best_aic_i]} (Best AIC)"),
        (best_bic_i, f"M = {values_of_M_to_try[best_bic_i]} (Best BIC)"),
        (-1, "M = 60 (Most complex)")
    ]
    # sample text from each model
    for idx, description in models_for_sampling:
        print(f"\n{description}:")
        sample_text = sample_hmm_text(models[idx], 500)
        print(sample_text)

    # QUESTION 2e to 2h
    print("\n Results (2e to 2h): ")

    # create noisy version of test text w/ random char erasures
    erase_rate = 0.2
    noisy_test_text, erase_mask = erase_random_chars(test_text, erase_rate)
    noisy_test_data = text_to_num(noisy_test_text)

    print(f"Original test text length: {len(test_text)}")
    print(f"Number of erased characters: {np.sum(erase_mask)}")
    print(f"Erasure rate: {np.sum(erase_mask)/len(test_text):.3f}")

    denoising_results = []

    for idx, description in models_for_sampling:
        print(f"\n Denoising with {description}:")

        # QUESTION 2g
        _, denoised_data = denoise_text(models[idx], noisy_test_data, noisy_test_text, erase_rate)
        denoised_text = num_to_text(denoised_data)

        # QUESTION 2h
        accuracy_erased = 0
        total_erased = 0
        # loop through test data
        for i in range(len(test_data)):
            # only check erased positions
            if erase_mask[i]:  
                total_erased += 1
                # check by comparing w/ ground truth
                if denoised_data[i] == test_data[i]:  
                    accuracy_erased += 1

        # calculate accuracy only if there were erased characters
        if total_erased > 0:
            accuracy = accuracy_erased / total_erased
        else:
            accuracy = 0.0
        print("Accuracy on erased characters: {:.3f} ({} out of {})".format(accuracy, accuracy_erased, total_erased))

        # print first 500 chars of denoised output
        print(f"First 500 characters of denoised text:")
        print(denoised_text[:500])

        # store results
        denoising_results.append((description, accuracy, denoised_text[:500]))

    # accuracy of randomly choosing one out of 27 characters
    rand_acc = 1.0 / 27
    print(f"\n Random choosing: {rand_acc:.3f}")