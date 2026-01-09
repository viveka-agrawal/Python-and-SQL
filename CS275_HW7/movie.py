
# UCI CS275P: Movie Rating Collaborative Filtering with Factor Analysis

import numpy as np
import matplotlib.pyplot as plt


from movie_utils import calc_rmse_for_movie_ratings, pca
from sparse_fa_em import sparse_fa_em

def fill_missing_ratings_with_mean(s):
    """
    Replace missing entries (0s) in each movie's ratings with the mean rating of that movie.

    Args:
      s: np.ndarray, shape (num_movies, num_users)
    Returns:
      s_filled: np.ndarray, shape (num_movies, num_users)
    """
    # make copy so original data is not overwritten
    s_filled = np.copy(s)

    # loop through each movie
    for mov_ind in range(s.shape[0]):
      # user ratings for current movie
      mov_rate = s[mov_ind, :]

      # find non-zero ratings
      obs = mov_rate != 0

      # calculate mean of observed ratings
      if np.sum(obs) > 0:
        mean_rate = np.mean(mov_rate[obs])

        # fill in missing ratings with mean
        for user_ind in range(s.shape[1]):
          if s_filled[mov_ind, user_ind] == 0:
            s_filled[mov_ind, user_ind] = mean_rate

    # return filled in matrix
    return s_filled

def reconstruct_from_latents(z, w, mu):
    """
    Reconstruct ratings from latent variables.

    Args:
      z : np.ndarray, shape (num_users, K)
      w : np.ndarray, shape (num_movies, K)
      mu: np.ndarray, shape (num_movies,)
    Returns:
      s_hat: np.ndarray, shape (num_users, num_movies)
    """
    # matrix mult btwn w and z transpose then add column vector mu
    s_hat = np.dot(w, z.T) + mu.reshape(mu.shape[0], 1)
    return s_hat

if __name__ == '__main__':
      # Load MovieLens dataset
      data = np.load("movielens500.npy", allow_pickle=True).item()
      s_train = data["S_train"].astype("float")
      s_test = data["S_test"].astype("float")

      num_movies = s_train.shape[0]
      num_users = s_train.shape[1]
      # to find k = {1,2,...,15}
      ks = range(1, 16)

      # Very naive baseline: Predict a rating of 3 for every missing rating
      # (This is NOT the same as the baseline you implement in fill_missing_ratings_with_mean.)
      s_hat = np.copy(s_train)
      s_hat[s_hat==0] = 3.0
      rmse = calc_rmse_for_movie_ratings(s_test, s_hat)
      print(rmse)

      # Example of performing dense PCA on interpolated data matrix
      # z_pca : #users x #movies, first K entries of row i give embedding for user i
      # w_pca : #movies x #movies, first K columns give best K-dim subspace
      # mu_pca: #movies x 1, gives mean rating subtracted before computing PCA eigenvectors
      z_pca, w_pca, mu_pca = pca(s_hat.T)

      # Example of training factor analysis with EM on true sparse data
      #  z_fa : #users x K, row i gives K-dim embedding for user i
      #  w_fa : #movies x K, factor loading matrix
      #  mu_fa: #movies x 1, mean rating for each movie (not just sample mean)
      #  psi  : #movies x 1, observation variance for each movie
      K = 3
      z_fa, w_fa, mu_fa, psi, _ = sparse_fa_em(s_train, K, max_iters=100, verbose=True)


      # Question 1a
      s_filled_mean = fill_missing_ratings_with_mean(s_train)
      baseline = calc_rmse_for_movie_ratings(s_test, s_filled_mean)
      print(f"Question 1a: RMSE = {baseline:.3f}")

      # Question 1b
      rmse_list_pca = []
      for k in ks:
          # run PCA on mean filled in training matrix
          z_full, w_full, mu_pca = pca(s_filled_mean)   # 500 movies, 943 users
          # keep only best components for users & movies
          z_pca = z_full[:, :k]  # for users: (943, k)
          w_pca = w_full[:, :k]  # for movies: (500, k)

          # reconstruct full rating matrix shape
          s_hat_pca = np.dot(w_pca, z_pca.T).T + mu_pca  # shape becomes (500, 943), same as s_test
          # make sure reconstructed matrix shape matches test set
          assert s_hat_pca.shape == s_test.shape, f"Shape mismatch: s_hat_pca {s_hat_pca.shape}, s_test {s_test.shape}"

          # calculate RMSE using test entries
          rmse = calc_rmse_for_movie_ratings(s_test, s_hat_pca)
          rmse_list_pca.append(rmse)
          print(f"Question 1b: PCA K = {k}: RMSE = {rmse:.3f}")

      # plot results
      plt.figure()
      plt.plot(ks, rmse_list_pca, label='PCA')
      plt.axhline(baseline, color='red', linestyle='--', label='Baseline')
      plt.xlabel('K')
      plt.ylabel('RMSE')
      plt.title('Question 1b: PCA RMSE vs K')
      plt.legend()
      plt.grid(True)
      plt.show()

      # Question 1d
      rmse_list_fa = []
      for k in ks:
          # train factor analysis model
          z_fa, w_fa, mu_fa, _, _ = sparse_fa_em(s_train, K=k, max_iters=100, verbose=False)
          # reconstruct full matrix
          s_hat_fa = reconstruct_from_latents(z_fa, w_fa, mu_fa)
          # calculate RMSE using test entries
          rmse = calc_rmse_for_movie_ratings(s_test, s_hat_fa)
          rmse_list_fa.append(rmse)
          print(f"Question 1d: FA K={k}: RMSE={rmse:.3f}")

      # find best K for FA
      best_k_fa = ks[np.argmin(rmse_list_fa)]
      print(f"Question 1d: Best FA performance at K = {best_k_fa}, RMSE = {min(rmse_list_fa):.3f}")

      # plot results
      plt.figure()
      plt.plot(ks, rmse_list_fa, label='Factor Analysis')
      plt.plot(ks, rmse_list_pca, label='PCA')
      plt.axhline(baseline, color='red', linestyle='--', label='Baseline')
      plt.xlabel('K')
      plt.ylabel('RMSE')
      plt.title('Question 1d: Factor Analysis vs PCA')
      plt.legend()
      plt.grid(True)
      plt.show()