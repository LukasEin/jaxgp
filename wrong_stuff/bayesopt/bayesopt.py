import numpy as np

def optimize_var_1D(X_init, Y_init, gpr, func, grid, *, noise=0.1, rel_std_noise = 1e-2, max_iters=10, return_fl=False):
    means_iter = []
    stds_iter = []
    features = X_init
    labels = Y_init

    max_std = np.infty

    for _ in range(max_iters):
        if np.abs(max_std) <= noise*rel_std_noise:
            break

        gpr.fit(features,labels)

        means, stds = gpr.predict(grid, return_std=True)
        means_iter.append(means)
        stds_iter.append(stds)

        max_std = np.max(stds)

        next_feature = grid[np.argmax(stds)]
        next_label = func(next_feature,noise)

        features = np.vstack((features,next_feature))
        labels = np.vstack((labels,next_label))

    if return_fl:
        return features, labels, np.array(means_iter), np.array(stds_iter)
    else:
        return np.array(means_iter), np.array(stds_iter)
    

def optimize_var_ND(X_init, Y_init, gpr, func, flattened_grid, *, noise=0.1, rel_std_noise = 1e-2, max_iters=10, return_fl=False):
    means_iter = []
    stds_iter = []
    features = X_init
    labels = Y_init

    max_std = np.infty

    for _ in range(max_iters):
        if np.abs(max_std) <= noise*rel_std_noise:
            break

        gpr.fit(features,labels)

        means, stds = gpr.predict(flattened_grid, return_std=True)
        means_iter.append(means)
        stds_iter.append(stds)

        avg_stds = np.mean(stds,axis=1)

        max_avg_std = np.max(avg_stds)

        next_feature = flattened_grid[np.argmax(avg_stds)]
        next_label = func(*next_feature,noise)

        features = np.vstack((features,next_feature))
        labels = np.vstack((labels,next_label))

    if return_fl:
        return features, labels, np.array(means_iter), np.array(stds_iter)
    else:
        return np.array(means_iter), np.array(stds_iter)