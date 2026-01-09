# UCI CS275P: Stochastic Gradient Descent for Logistic Regression 
# Data: MNIST Digits 4 & 9

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Imports to use for optimization
from functools import partial
import scipy.optimize

# Homework Imports
from mnistSGD_utils import *

def min_func_sgd(objective_func, objective_func_grad, w0, batches, options):
    '''
    Minimizes using SGD

    Parameters:
        objective_func: The function being optimized
        objective_func_grad: The gradient function that computes the gradient of the function being optimized
        w0: The initial parameters
        batches: The batches of data to use
        options: Dict of options for the optimizer	
                        - max_epoch: max number of epochs to use
                        - max_updates: max number of update steps to take
                        - step_size_fn: function that takes in the current step number and returns the step size to take
                        - verbose: If we should be verbose or not

    Returns:
        w_history: A list containing the weights w at each epoch
    '''

    # Extract the options
    max_epoch = options["max_epoch"]
    max_updates = options["max_updates"]
    step_size_fn = options["step_size_fn"]
    verbose = options["verbose"]

    # The w we will optimize
    w = np.copy(w0)
    w_history = [np.copy(w)] # Store initial weights

    total_num_updates = 0
    for epoch in range(max_epoch):
        # TODO: Implement the SGD loop
        
        # loop through batches in current epoch
        for b in batches:
            # obtain data & labels for current batch
            x_batch, y_batch = b  
            # calculate step size
            step_size = step_size_fn(total_num_updates)  
            # calculate gradient
            gradient = objective_func_grad(w, x_batch, y_batch)  
            # update weights
            w = w - (step_size * gradient) 
            # increment total number of updates
            total_num_updates += 1  

            # check if max number of updates is reached
            if total_num_updates >= max_updates:
                break

        # Store weights at the end of the epoch
        w_history.append(np.copy(w))

        if(verbose):
            # Print some stats for the user
            if(((epoch % 10) == 0) or (epoch == (max_epoch-1))):
                print("  epoch {:4d} | update {:6d}/{:6d} | {:6d} batches | {:.4f} step size".format(epoch, total_num_updates, max_updates, len(batches), step_size));

            # Check if we hit our max number of updates
        if(total_num_updates >= max_updates):
            break

    return w_history # Return the history of weights


def main():
    # If we should save the plots or not
    do_save_plots = True

    # Load the data
    data = np.load("MNIST_Digits49_19x19.npy", allow_pickle=True).item()
    x_train = data["Xtrain"]
    y_train = data["ytrain"]
    x_test = data["Xtest"]
    y_test = data["ytest"]

    # Extract some information
    num_dims = x_train.shape[-1]


    ################################################################################
    # FULL-DATASET gradient descent (part a)
    ################################################################################
    print("Full dataset gradient descent:")

    # Create the partial functions that we will be using for the optimization.
    # These functions bind the loss and gradient functions such that we can just 
    # pass in a value for w and evaluate the function.
    # You must derive and implement the gradient function, NOT approximate numerically.
    loss_func = partial(logistic_loss_scaled, X=x_train, y=y_train)
    grad_func = partial(logistic_loss_scaled_grad, X=x_train, y=y_train)

    # Optimize using the scipy optimization library
    # Here we use the BFGS optimization method which takes into account the history
    # of gradient evaluations during optimization to accelerate convergence.
    options = dict()
    options["maxiter"] = 20000
    options["ftol"] = 1e-7
    w0 = np.zeros((num_dims,))
    start_time = get_current_time_milliseconds()
    results = scipy.optimize.minimize(fun=loss_func, x0=w0, jac=grad_func, \
        method="L-BFGS-B", options=options)
    elapsed_time = get_current_time_milliseconds() - start_time
    assert(results.success) # make sure it converged
    w_min = results.x  # the value for w that minimizes the loss func

    # Calculate and print Stats
    train_acc_full = calc_log_reg_accuracy(w_min, x_train, y_train)
    test_acc_full  = calc_log_reg_accuracy(w_min, x_test, y_test)
    print("  {:.3f} msec elapsed".format(elapsed_time))
    print("  accuracy Train {:.3f}".format(train_acc_full))
    print("  accuracy Test  {:.3f}".format(test_acc_full))
    print("")

    ################################################################################
    # STOCHASTIC GRADIENT DESCENT (parts b-c)
    ################################################################################

    # Maximum number of passes through the full dataset
    max_epoch = 30

    # The different parameter settings to try
    kappa_list = [0.51, 0.7]
    num_batches_list = [1, 100, 11791]

    # Define a step size function \eta_t = (10 + t)^{-kappa}
    def step_size_func(step_number, kappa_in):
        return np.power(step_number+10, -kappa_in)

    # Define callback function for assessing accuracy after each epoch of SGD
    callback_funcs = []
    callback_funcs.append(partial(calc_log_reg_accuracy, X=x_train, y=y_train))
    callback_funcs.append(partial(calc_log_reg_accuracy, X=x_test, y=y_test))


    # Plot settings
    colors = ["blue", "red", "black", "magenta"]
    styles = ["-", "--", "-."]

    for kk in range(len(kappa_list)):
        for bb in range(len(num_batches_list)):
            kappa = kappa_list[kk]
            num_batches = num_batches_list[bb]

            # Create the batches of the training data
            training_batches = create_batches_from_full_dataset(x_train, y_train, num_batches)

            # Config sgd options
            sgdopts = dict()
            sgdopts["max_epoch"] = max_epoch
            num_updates_per_epoch = num_batches
            sgdopts["max_updates"] = int(num_updates_per_epoch * max_epoch)
            sgdopts["step_size_fn"] = partial(step_size_func, kappa_in=kappa)
            sgdopts["verbose"] = False

            # Estimate weights via SGD
            w0 = np.zeros((num_dims,))
            start_time = get_current_time_milliseconds()
            # Run the SGD optimizer - now returns history
            print(f"Running SGD for Batches:{num_batches}, Kappa:{kappa}...")
            w_history = min_func_sgd(
                logistic_loss_scaled, 
                logistic_loss_scaled_grad, 
                w0, 
                training_batches, 
                sgdopts)
            elapsed_time = get_current_time_milliseconds() - start_time
            print("SGD Finished.")

            # Determine the number of epochs actually completed
            num_completed_epochs = len(w_history) - 1 # -1 because history includes w0

            # Compute accuracy only at the end of each completed epoch
            # w_history contains [w_initial, w_epoch_0, w_epoch_1, ..., w_epoch_{num_completed_epochs-1}]
            # Accuracy after epoch `epoch` (0-indexed) corresponds to index `epoch + 1`
            epoch_train_acc = [calc_log_reg_accuracy(w_history[epoch + 1], x_train, y_train) for epoch in range(num_completed_epochs)]
            epoch_test_acc = [calc_log_reg_accuracy(w_history[epoch + 1], x_test, y_test) for epoch in range(num_completed_epochs)]

            print("Stoch Grad Descent | nBatch {:d}, Kappa {:f}".format(num_batches, kappa))
            print("  {:.3f} msec elapsed".format(elapsed_time))
            # Print final accuracy (accuracy corresponding to the last completed epoch)
            if num_completed_epochs > 0:
                print("  final accuracy Train {:.3f}".format(epoch_train_acc[-1]))
                print("  final accuracy Test {:.3f}".format(epoch_test_acc[-1]))
            else:
                # Handle case where no epochs were completed (e.g., max_updates=0 or very small)
                initial_train_acc = calc_log_reg_accuracy(w_history[0], x_train, y_train)
                initial_test_acc = calc_log_reg_accuracy(w_history[0], x_test, y_test)
                print("  initial accuracy Train {:.3f}".format(initial_train_acc))
                print("  initial accuracy Test {:.3f}".format(initial_test_acc))
            print()

            # Plot accuracy history vs epoch
            epochs = list(range(num_completed_epochs)) # Use actual completed epochs
            if num_completed_epochs > 0: # Only plot if at least one epoch completed
                fig1 = plt.figure(1)
                plt.plot(epochs, epoch_train_acc, color=colors[bb], linestyle=styles[kk], \
                    label="Batches:{:d}, Kappa:{:0.3f}".format(num_batches, kappa))

                fig2 = plt.figure(2)
                plt.plot(epochs, epoch_test_acc, color=colors[bb], linestyle=styles[kk], \
                    label="Batches:{:d}, Kappa:{:0.3f}".format(num_batches, kappa))


    split_names = ["Train", "Test"]
    for f in [1, 2]:

        fig = plt.figure(f)
        plt.title("{} Accuracy vs Epoch".format(split_names[f-1]))
        plt.xlabel("Epoch Number (Num. passes thru data)")
        plt.ylabel("Accuracy")

        if(f == 1):
            plt.axhline(y=train_acc_full, color="magenta", linestyle="--", \
                label="Full Data (L-BFGS-B)")
        else:
            plt.axhline(y=test_acc_full, color="magenta", linestyle="--", \
                label="Full Data (L-BFGS-B)")

        plt.legend()


    if(do_save_plots):
        fig1.savefig("accVsEpoch_train.pdf")
        fig2.savefig("accVsEpoch_test.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()