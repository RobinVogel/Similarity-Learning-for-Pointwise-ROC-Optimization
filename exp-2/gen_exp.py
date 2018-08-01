import numpy as np
import math
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd

import utils as ut

# --- Binary classification ---

def eta_a(x, a=1.):
    """ See the math in the notebook. """
    a = float(a)
    k = a / (1 - a)
    if x >= 0.5:
        return 0.5 + 0.5 * math.pow(2. * x - 1., (1. / k))
    else:
        return 0.5 - 0.5 * math.pow(1. - 2. * x, (1. / k))


def datagen_eta_a(n=100, a=1.):
    """ Generate data for X ~ U[0,1] and eta_a. """
    X = np.random.uniform(0., 1., n)
    Y = 2 * np.random.binomial(n=1, p=list(map(lambda x: eta_a(x, a), X))) - 1
    return X, Y


def classif_error(f, a=1.):
    """ Gives the theoretical error given a frontier f that separates positives and negatives. """
    a = float(a)
    return (1 - a) / 2 + (a / 2) * math.pow(abs(2 * f - 1), 1 / a)


def classification_experiments(n_list, a_collection, n_exps=1000, verbose=True):
    """
    Returns a DataFrame with the results for 1000 experiences 
    for each n in n_list and each a in a_collection.
    """
    d = {"a": [], "n": [], "gen_error": [], "frontier": []}
    for i in range(0, n_exps):
        for a_test in a_collection:
            # Do the experiments for the collection of k's
            n_tot = np.max(n_list)
            X, Y = datagen_eta_a(n=n_tot, a=a_test)
            for n_cur in n_list:
                # Do the experiments for the collection of n's
                f_opt, _, _ = ut.bipart_partition(X[0:n_cur], Y[0:n_cur], criterion = lambda np, nn : np - nn, epsilon = 0.001) 
                err_est = classif_error(f_opt, a=a_test)
                err_theo = classif_error(0.5, a=a_test)
                # Save results 
                d["a"] += [a_test]
                d["n"] += [n_cur]
                d["gen_error"] += [err_est - err_theo]
                d["frontier"] += [f_opt]
        if verbose and i % 50 == 0:
            print("Done {:3} %".format(int((i / n_exps)*100)))
    return pd.DataFrame(d)


# --- Bipartite ranking ---


def bipart_interval_alpha(a, m=0.25):
    """
    Returns an interval that alpha has to belong to to respect 
    the conditions C >= 0 and C <= 1/2.
    """
    inter = np.power(1 - 2 * m, 1 / a)
    return (1 - 2 * m - a * inter) / 2, (1 - a * inter) / 2


def bipart_C(a, alpha, m=0.25):
    """ Returns the C for a given alpha and a in the bipartite ranking scheme. """
    if m > 0.:
        return 0.5 + (2 * alpha - 1 + a * np.power(1 - 2 * m, 1 / a)) / (4 * m)
    else:
        return 0.


def eta_aCm(x, a, C, m=0.25):
    """ See the math in the notebook. """
    if x < 0:
        return None
    elif x < m:
        return C
    elif x < 1 - m:
        return eta_a(x, a=a)
    elif x <= 1:
        return 1 - C


def datagen_eta_aCm(n, a, C=None, m=0.25):
    """ Generate data for X ~ U[0,1] and eta_aCm. """
    if C is None:
        print("Unspecified C, choosen with bipart_C.")
        C = bipart_C(a, alpha)
    X = np.random.uniform(0., 1., n)
    Y = 2 * np.random.binomial(n=1,
                               p=list(map(lambda x: eta_aCm(x, a, C, m=m), X))) - 1
    return X, Y


def int_eta_a(x, a):
    """ Integral of eta_aCm over [0,x]. """
    return x / 2 + (a / 4) * (np.power(abs(1 - 2 * x), 1 / a) - 1)


def birank_G_aCm(f, a, C=None, m=0.25, alpha=0.15):
    """ Computes G for a frontier in the bipartite ranking scheme. """
    a = float(a)
    if C is None:
        C = bipart_C(a, alpha, m=m)
    res = 0
    if f >= 0.0:
        res += C * min(f, m)
    if f >= m:
        val_f = min(f, 1. - m)
        res += int_eta_a(val_f, a=a) - int_eta_a(m, a=a)
    if f >= 1. - m:
        res += (1 - C) * (min(1., f) - (1 - m))
    return 2 * (1. / 2 - res)


def birank_H_aCm(f, a, C=None, m=0.25, alpha=0.15):
    """ Computes H for a frontier f in the bipartite ranking scheme. """
    return 2 * (1. - f) - birank_G_aCm(f, a, C=C, m=m, alpha=alpha)

def exp_bi_ranking(n_list, n_exps, a_collection, alpha=0.38, m=0.25):
    """ 
    Does the experiments associated to the bipartite ranking problem.
    """
    t_0 = time.time()

    dict_res = dict()
    n_tot = np.max(n_list)

    for i in range(0, n_exps):
        for a in a_collection:
            a = float(a)
            if m <= 0:
                alpha = (1 - a) / 2
                C = 0
                m = 0
            else:
                C = bipart_C(a, alpha, m)
            X, Y = datagen_eta_aCm(n_tot, a, C=C, m=m)
            G_theo = birank_G_aCm(0.5, a, C=C, m=m, alpha=alpha)
            H_theo = birank_H_aCm(0.5, a, C=C, m=m, alpha=alpha)

            for n_cur in n_list:
                X_cand = X[0:n_cur]
                Y_cand = Y[0:n_cur]
                n_p = (Y_cand == +1).sum()
                n_n = X_cand.shape[0] - n_p

                def opt_fun(i_p, i_n):
                    if float(i_n) / n_n <= alpha:
                        return i_p
                    else:
                        return - float("inf")
                f_opt, _, _ = ut.bipart_partition(X_cand, Y_cand, opt_fun)
                H_est = birank_H_aCm(f_opt, a, C=C, m=m, alpha=alpha)
                G_est = birank_G_aCm(f_opt, a, C=C, m=m, alpha=alpha)
                phi_nd = (H_est - alpha) / 2

                d_update = {"a": a, "n": n_cur, "phi_nd": phi_nd,
                    "GR": G_theo, "GRn": G_est,
                    "HR": H_theo, "HRn": H_est,
                    "frontier": f_opt, "gen_error": G_theo - G_est}

                dict_res = { k : dict_res.get(k, []) + [d_update[k]] for k in d_update.keys() }
        if i % 50 == 0:
            print("Done {:3} %".format(int(100*(i/n_exps))))
    return pd.DataFrame(dict_res)


# --- Similarity ranking ---

def illustrate_As(m=0.25):
    """ Illustrates the decomposition of the integral over A1, A2, A3 """
    def return_plot(x0, x1, y0, y1):
        return [x0] * 2 + [x1] * 2 + [x0], [y0] + [y1] * 2 + [y0] * 2
    xX, yX = return_plot(0, 1., 0, 1.)
    plt.plot(xX, yX, label="pair space")
    xA1, yA1 = return_plot(1 - m, 1, 1 - m, 1)
    plt.plot(xA1, yA1, color="orange", linewidth=3, label="A1")
    xA3, yA3 = return_plot(0.5, 1 - m, 0.5, 1 - m)
    plt.plot(xA3, yA3, color="red", linewidth=3, label="A3")
    xA2, yA2 = return_plot(1 - m, 1, 0.5, 1 - m)
    plt.plot(xA2, yA2, color="green", linewidth=3, label="A2")
    plt.plot([0, 1], [0, 1], "b--", label="symmetry eta")
    plt.plot([1, 0], [0, 1], "b--")
    plt.xlabel("x")
    plt.ylabel("x'")
    plt.title("")
    plt.legend(loc="lower left")
    plt.show()

def simrank_eta(x, xp, a=0.4, alpha=0.38, m=0.25):
    """ See the math in the notebook. """
    C = simrank_C(a, alpha, m=m)
    eta_x = eta_aCm(x, a, C, m=m)
    eta_xp = eta_aCm(xp, a, C, m=m)
    return 0.5 + 0.5 * (2 * eta_x - 1) * (2 * eta_xp - 1)


def simrank_interval_alpha(a, m):
    """
    Returns an interval that alpha has to belong to to respect 
    the conditions C >= 0 and C <= 1/2.
    """
    v1 = (2 * m + a * np.power(1 - 2 * m, 1. / a))**2
    v2 = ((a**2) * np.power(1 - 2 * m, 2. / a)) / (4 * (m**2))
    return 0.5 - v1 / 2, 0.5 - v2 / 2


def simrank_C(a, alpha, m=0.25):
    """ Returns the C for a given alpha and a in the similarity ranking scheme. """
    if m > 0:
        num = a * np.power(1 - 2 * m, 1. / a) + 2 * m - np.sqrt(1 - 2 * alpha)
        denom = 4 * m
        return num / denom
    else:
        return 0.
        

def int_fm1(min_x0, max_x0, a, m, C):
    """
    Assumes min_x0 < max_x0.
    Computes the integral of f(x)-1 over min_x0, max_x0 which is
    an intermediary step before computing the integral of eta over rectangles.
    """
    res = 0
    deb_reg = min(max(min_x0, 0.), m)
    end_reg = min(max_x0, m)
    res += (end_reg - deb_reg) * (2 * C - 1)
    # f-1 = 2 \eta - 1
    if max_x0 > m:
        deb_reg = min(max(min_x0, m), 1. - m)
        end_reg = min(max_x0, 1. - m)
        res += 2 * (int_eta_a(end_reg, a=a) -
                    int_eta_a(deb_reg, a=a)) - (end_reg - deb_reg)
    if max_x0 > 1. - m:
        deb_reg = min(max(min_x0, 1. - m), 1.)
        end_reg = min(max_x0, 1.)
        res += (end_reg - deb_reg) * (1 - 2 * C)
    return res


def square_int(min_x, min_xp, max_x, max_xp, a, m, C):
    """
    Computes the integral of eta over rectangles.
    """
    intf1 = int_fm1(min_x, max_x, a, m, C)
    intf2 = int_fm1(min_xp, max_xp, a, m, C)
    return 0.5 * (max_x - min_x) * (max_xp - min_xp) + 0.5 * intf1 * intf2

def simrank_G_aCm(f, a, C=None, m=0.25, alpha=0.15):
    """ Computes G for a frontier f in the similarity ranking scheme. """
    a = float(a)
    if C is None:
        C = simrank_C(a, alpha, m=m)
    res = 0
    if f >= 0. and f < 0.5:
        val_f = min(f, 0.5)
        res = 2 * square_int(0., 0., val_f, val_f, a, m, C)
    if f >= 0.5:
        val_f = min(f, 1.)
        # Integ de 0.5 a f
        res = square_int(0., 0., 1., 1., a, m, C) - 2 * \
            square_int(0., val_f, 1. - f, 1., a, m, C)
    return res / 0.5


def simrank_H_aCm(f, a, C=None, m=0.25, alpha=0.15):
    """ Computes H for a frontier f in the similarity ranking scheme. """
    int_eta = simrank_G_aCm(f, a, C=C, m=m, alpha=alpha)
    if f < 0.5:
        return 4 * (f**2) - int_eta
    else:
        return 2 * (1 - 2 * ((1 - f)**2)) - int_eta


def optimum_simrank(x_p, x_n, alpha):
    # Intermediary function to the one below.
    n_p = x_p.shape[0]
    n_n = x_n.shape[0]
    
    pos_pair_1 = itertools.combinations(x_p, 2)
    pos_pair_2 = itertools.combinations(x_n, 2)
    neg_pair = itertools.product(x_p, x_n)

    def get_val_from_pair(x):
        # Transforms each pair into one minus the minimum of its l1 distance to (0,0) or (1,1).
        distance_to_lower_corner = max(abs(x[0]), abs(x[1]))
        distance_to_upper_corner = max(abs(1. - x[0]), abs(1. - x[1]))
        return 1 - min(distance_to_lower_corner, distance_to_upper_corner)
    
    x_p = np.array(list(map(get_val_from_pair, pos_pair_1))+ list(map(get_val_from_pair, pos_pair_2)))
    x_n = np.array(list(map(get_val_from_pair, neg_pair)))

    def opt_fun(i_p, i_n):
        if float(i_n) / x_n.shape[0] <= alpha:
            return i_p / x_p.shape[0]
        else:
            return - float("inf")
    
    X = np.hstack([x_p,x_n])
    Y = np.array([ +1 ]*len(x_p) + [ -1 ]*len(x_n))
    f_opt, crit_opt, _ = ut.bipart_partition(X, Y, opt_fun)
    
    return 1-f_opt, crit_opt


def exp_sim_ranking(n_list, n_exps, a_collection, m=0.25, alpha=0.15):
    """
    Does the experiments that are required for pointwise ROC optimization for similarity ranking.
    """
    print("Starting...")
    t_0 = time.time()
    dict_res = dict()

    n_tot = np.max(n_list)
    c_time = time.time()
    for i in range(0, n_exps):
        for a in a_collection:
            a = float(a)
            if m <= 0:
                C = 0
                alpha = (1 - a**2) / 2
                m = 0
            else:
                C = simrank_C(a, alpha, m)
            X, Y = datagen_eta_aCm(n_tot, a, C=C, m=m)
            G_theo = simrank_G_aCm(0.5, a, C=C, m=m, alpha=alpha)
            H_theo = simrank_H_aCm(0.5, a, C=C, m=m, alpha=alpha)

            for n_cur in n_list:
                X_cand = X[0:n_cur]
                Y_cand = Y[0:n_cur]
                x_p = X_cand[Y_cand == +1]
                x_n = X_cand[Y_cand == -1]
                f_opt, crit_opt = optimum_simrank(x_p, x_n, alpha=alpha)
                
                G_est = simrank_G_aCm(f_opt, a, C=C, m=m, alpha=alpha)
                H_est = simrank_H_aCm(f_opt, a, C=C, m=m, alpha=alpha)
                phi_nd = (H_est - alpha) / 2

                d_update = {"a": a, "n": n_cur, "phi_nd": phi_nd,
                    "G_theo": G_theo, "G_est": G_est,
                    "H_theo": H_theo, "H_est": H_est,
                    "f_opt": f_opt, "gen_error": G_theo - G_est}
                    
                dict_res = { k : dict_res.get(k, []) + [d_update[k]] for k in d_update.keys() }

        if i % 10 == 0:
            print("Done {:3}.... {:3.0f} s".format(int(100*(i/n_exps)), time.time() - c_time))

    return pd.DataFrame(dict_res)

if __name__ == "__main__":
    prefix_path = ""
    n_exps = 42 
    m = 0.255
    alpha = 0.37
    n_list = [ int(np.power(2.,i)) for i in range(6, 10)] 
    a_collection = np.linspace(0.1, 0.9, 15) # 0.9, 9)
    df_simrank_alpha = exp_sim_ranking(n_list, n_exps, a_collection, m=m, alpha=alpha)
    df_simrank_alpha.to_csv(prefix_path + "simrank"+str(sys.argv[1])+".csv", index=None)