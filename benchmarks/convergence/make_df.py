import jax.numpy as jnp
from jax import vmap

import pandas as pd

def mse(Y, prediction):
    return jnp.mean((Y-prediction)**2)

def maxerr(Y, prediction):
    return jnp.max(jnp.absolute(Y-prediction))

def maxstd(std):
    return jnp.max(std)

def true_in_conf(Y, prediction, std):
    inside = jnp.where(jnp.less(Y,prediction+std) * jnp.greater(Y,prediction-std), 1.0, 0.0)
    return jnp.mean(inside)

def make_dict(num_f_vals, num_d_vals, optimizer, fun, eval_grid, predictions, stds):
    Y = fun(eval_grid)

    true_in_confs = []
    for std, pred in zip(stds, predictions):
        true_in_confs.append(true_in_conf(Y, pred, std))

    temp = {"f": num_f_vals, "d": num_d_vals, "opt": optimizer, 
            "mses": vmap(mse, in_axes=(None, 0))(Y, predictions),
            "maxerrs": vmap(maxerr, in_axes=(None, 0))(Y, predictions),
            "maxstds": vmap(maxstd, in_axes=(0,))(stds),
            "true_in_confs": jnp.array(true_in_confs)}
    
    mses = {"mean_mse": jnp.mean(temp["mses"]), 
         "max_mse": jnp.max(temp["mses"]),
         "min_mse": jnp.min(temp["mses"])}
    
    confs = {"mean_tic": jnp.mean(temp["true_in_confs"]), 
         "max_tic": jnp.max(temp["true_in_confs"]),
         "min_tic": jnp.min(temp["true_in_confs"])}
    
    return {**temp, **mses, **confs}

def make_dict_all(num_vals, optimizer, fun, eval_grid, predictions, stds):
    Y = fun(eval_grid)

    true_in_confs = []
    for std, pred in zip(stds, predictions):
        true_in_confs.append(true_in_conf(Y, pred, std))

    temp = {"f": num_vals, "opt": optimizer, 
            "mses": vmap(mse, in_axes=(None, 0))(Y, predictions),
            "maxerrs": vmap(maxerr, in_axes=(None, 0))(Y, predictions),
            "maxstds": vmap(maxstd, in_axes=(0,))(stds),
            "true_in_confs": jnp.array(true_in_confs)}
    
    mses = {"mean_mse": jnp.mean(temp["mses"]), 
         "max_mse": jnp.max(temp["mses"]),
         "min_mse": jnp.min(temp["mses"])}
    
    confs = {"mean_tic": jnp.mean(temp["true_in_confs"]), 
         "max_tic": jnp.max(temp["true_in_confs"]),
         "min_tic": jnp.min(temp["true_in_confs"])}
    
    return {**temp, **mses, **confs}

def make_df(list_f_vals, list_d_vals, optimizers, in_dir, name, sparse, subset_size, fun, eval_grid):
    in_list = []

    for num_f_vals in list_f_vals:
        for num_d_vals in list_d_vals:        
            for optimizer in optimizers:
                if sparse:
                    fname = f"{in_dir}/{name}_d{num_d_vals}_f{num_f_vals}_{optimizer}_sparse{subset_size}"

                    means = jnp.load(f"{fname}_means.npz")
                    stds = jnp.load(f"{fname}_stds.npz")
                else:
                    fname = f"{in_dir}/{name}_f{num_f_vals}d{num_d_vals}"

                    means = jnp.load(f"{fname}means{optimizer}.npz")
                    stds = jnp.load(f"{fname}stds{optimizer}.npz")
                
                means_list = []
                for key, value in means.items():
                    means_list.append(value)
                stds_list = []
                for key, value in stds.items():
                    stds_list.append(value)

                # for i, (mean, std) in enumerate(zip(means, stds)):
                #     is_nan = jnp.any(jnp.isnan(mean)) or jnp.any(jnp.isnan(mean))
                #     if is_nan:
                #         print(f"Nan values at f={num_f_vals}, d={num_d_vals}, opt={optimizer}, iter={i}!")
                means = jnp.array(means_list)
                stds = jnp.array(stds_list)

                in_list.append(make_dict(num_f_vals, num_d_vals, optimizer, fun, eval_grid, means, stds))

    return pd.DataFrame(in_list)

def make_df_all(list_vals, optimizers, in_dir, name, sparse, subset_size, fun, eval_grid):
    in_list = []

    for num_vals in list_vals:        
        for optimizer in optimizers:
            if sparse:
                fname = f"{in_dir}/{name}_v{num_vals}_{optimizer}_sparse{subset_size}"

                means = jnp.load(f"{fname}_means.npz")
                stds = jnp.load(f"{fname}_stds.npz")
            else:
                fname = f"{in_dir}/{name}_v{num_vals}"

                means = jnp.load(f"{fname}means{optimizer}.npz")
                stds = jnp.load(f"{fname}stds{optimizer}.npz")
            
            means_list = []
            for key, value in means.items():
                means_list.append(value)
            stds_list = []
            for key, value in stds.items():
                stds_list.append(value)

            # for i, (mean, std) in enumerate(zip(means, stds)):
            #     is_nan = jnp.any(jnp.isnan(mean)) or jnp.any(jnp.isnan(mean))
            #     if is_nan:
            #         print(f"Nan values at f={num_f_vals}, d={num_d_vals}, opt={optimizer}, iter={i}!")
            means = jnp.array(means_list)
            stds = jnp.array(stds_list)

            in_list.append(make_dict_all(num_vals, optimizer, fun, eval_grid, means, stds))

    return pd.DataFrame(in_list)