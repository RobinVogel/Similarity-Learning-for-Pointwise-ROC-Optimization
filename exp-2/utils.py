import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

# --- Preliminaries ---

# - Partitioning techniques -
class BestSolutionHolder:
    """
    Keeps in memory the best solution of a partitioning technique, 
    is used in the function below.
    """
    def __init__(self, crit, n_pos, n_neg, frontier, orient, 
                 left_n_right=False, n_pos_tot=None, n_neg_tot=None):
        self.crit = crit
        self.left_n_right = left_n_right
        if left_n_right:
            if n_pos_tot is None or n_neg_tot is None:
                raise ValueError("Please indicate the total number of positives or negatives.")
            self.n_pos_tot = n_pos_tot
            self.n_neg_tot = n_neg_tot
            
        self.best_val = None
        self.best_front = None
        self.best_orient = None
        
        self.calls = 0
            
        self.save_sol(n_pos, n_neg, frontier, orient)
                    
    def save_sol(self, n_pos, n_neg, frontier, orient):
        self.calls += 1
        
        val = self.crit(n_pos, n_neg)
        
        if self.best_val is None or val > self.best_val:
            self.best_val = val
            self.best_front = frontier
            self.best_orient = orient
            
        if self.left_n_right:
            val = self.crit(self.n_pos_tot - n_pos, self.n_neg_tot - n_neg)
            
            if self.best_val is None or val > self.best_val:
                self.best_val = val
                self.best_front = frontier
                self.best_orient = not(orient)    

def bipart_partition(X, Y, criterion, epsilon = 0.001):
    """
    Takes values in R associated to a class in {-1, +1}, a criterion that depends solely
    on the number of positive or negative values in a partition and finds the partition 
    that maximizes the criterion.
    """
    # crit = lambda np, nn : np - nn
    n = X.shape[0]
    n_pos_tot = (Y == +1).sum()
    n_neg_tot = n - n_pos_tot
    XY = np.vstack([X,Y]).transpose()
    sorted_XY = XY[np.argsort(XY[:,0])]
    n_pos, n_neg = 0, 0
    best_value = BestSolutionHolder(criterion, n_pos, n_neg, sorted_XY[0,0]-epsilon, False, 
                                    left_n_right=True, n_pos_tot=n_pos_tot, n_neg_tot=n_neg_tot)
    i = 0
    for xy, xyp in zip(sorted_XY, sorted_XY[1:]):
        x, y = xy
        xp, yp = xyp
        frontier = (x + xp)/2 
        # assumed the density of X continuous hence xp > x
        n_pos += int(y == +1)
        n_neg += int(y == -1)
        best_value.save_sol(n_pos, n_neg, frontier, False)
        i += 1

    frontier = xp + epsilon
    n_pos += int(yp == +1)
    n_neg += int(yp == -1)
    best_value.save_sol(n_pos, n_neg, frontier, False)
    return best_value.best_front, best_value.best_val, best_value.best_orient

# - Plotting utils -

def plot_eta(a_collection, param_fun, n_points=100):
    """
    Plots a family of functions defined on [0,1] indexed by a
    for a set of possible values for a.
    """
    plt.figure(figsize=(4, 4))

    x_ax = np.linspace(0., 1., n_points)

    for a in a_collection:
        eta_fun = lambda x: param_fun(x, a=a)
        plt.plot(x_ax, list(map(eta_fun, x_ax)),
                 label="$a = {:0.2f}$".format(a))

    plt.title("The function $\eta_a$ for different a")
    plt.xlabel("x")
    plt.ylabel("$\eta_a$(x)")
    plt.grid()
    plt.legend()
    plt.show()

def plot_distribution(a_collection, gen_datafun):
    """
    Plots the distributions induced by posterior probability functions defined on [0,1]
    indexed by a for a set of possible values for a.
    """
    plt.figure(figsize=(8, 10))
    n_a_col = len(a_collection)

    for i, a in enumerate(a_collection):
        plt.subplot(5, 2, i + 1)

        X, Y = gen_datafun(n=500, a=a)
        plt.grid()
        plt.hist(X[Y == -1], bins=30, color="red", normed=True, alpha=0.5)
        plt.hist(X[Y == 1], bins=30, color="green", normed=True, alpha=0.5)
        plt.title("Sample distribution, $a = {:2.2f}$".format(a))
    plt.tight_layout()
    plt.show()
    
def GetSlope(df_int, quant_val=0.9):
    """
    Gets the slope corresponding to the log of a quantile of
    the generalization error regressed by the logarithm of n, number of data points.
    Parameters:
    * df_int: Dataframe containing the columns "n" and "gen_error" 
        with enough entries for each value of "n".
    * quant_val: quantile value that we choose.
    Returns:
    * constant, slope: values in R.
    """
    groupby_quant = df_int.groupby("n").quantile(quant_val)
    vals_med = groupby_quant["gen_error"].values
    ns = df_int["n"].unique()
    reg = scipy.stats.linregress(np.log(ns), np.log(vals_med))
    return np.exp(reg.intercept), reg.slope

def boxplot_slopes(df, quant=0.5, ylim=[10**(-4), 10**(0)]):
    """
    Does a boxplot for each a of the results, to show the different generalization speeds.
    Parameters:
    * df: Dataframe containing the columns "a", "n" and "gen_error" 
        with enough entries for each values of "n","a".
    * quant_val: quantile value that we choose to regress the empirical generalization speed.
    """
    plt.figure(figsize=(20, 12))

    x_n_plots = len(df["a"].unique())//2 + 1
    for i, a in enumerate(df["a"].unique()):
        df_int = df[df["a"] == a]
        ns = df_int["n"].unique()
        bxplt_vals = [df_int[df_int["n"] == n_val]["gen_error"].values 
            for n_val in df_int["n"].unique()]

        correct_width = [5] + list(map(lambda x: x / 4, ns[1:]))
        plt.subplot(x_n_plots, 4, i + 1)
        plt.boxplot(bxplt_vals, positions=ns,
                    widths=correct_width)
        plt.xscale("log")
        plt.yscale("log")

        reg_med_const, reg_med_slope = GetSlope(df_int, quant_val=quant)

        plt.plot(ns, reg_med_const * np.power(ns, reg_med_slope), '-b')

        theo_slope = -1. / (2. - a)
        plt.plot(ns, 10**(theo_slope * np.log10(ns)), '-g')

        plt.title(
            "Experiment $a = {:1.1f}$\nSlope med (blue): ${:0.2f}$\nTheoretical bound slope (green) : ${:0.2f}$".format(
                a, reg_med_slope, theo_slope))
        plt.ylim(ylim)
        
        plt.xlabel("$n$")
        plt.ylabel("Regret")
        plt.grid()
        
    plt.tight_layout()

    plt.show()


def slopes_quants(df, quant_values=np.linspace(0.7, 0.9, 5)):
    """
    Compares the theoretical generalization slope and the empirical generalization slope.
    Parameters:
    * df: Dataframe containing the columns "a", "n" and "gen_error" 
        with enough entries for each values of "n","a".
    * quant_values: quantile values that we choose to regress the empirical generalization speed.
    """
    plt.figure(figsize=(8, 15))
    a_collection = df["a"].unique()
    
    theo_slopes = [-1. / (2. - a) for a in a_collection]
    emp_vals = []
    
    for i, quant in enumerate(quant_values):
        reg_med_slopes = list()
        for a in a_collection:
            df_int = df[df["a"] == a]
            reg_med_slopes.append(GetSlope(df_int, quant_val=quant)[1])
        emp_vals.append(reg_med_slopes)
        plt.subplot(5, 2, i + 1)
        plt.plot(theo_slopes, reg_med_slopes, "bo")
        plt.plot([-1, -0.5], [-1, -0.5], color="red")
        plt.xlabel("Theoretical slopes")
        plt.ylabel("Experimental slopes\n(quantile at ${:2.2f}$)".format(quant))
        plt.grid()
        
    plt.tight_layout()

    plt.show()

    
def plot_select_eta(a_collection, eta_fun_of_a, datagen_fun_of_a, n=1000, n_points=100, n_bins=20):
    """
    Plots the empirical distribution of the data, as well as the theoretical distribution of it.
    """
    x_ax = np.linspace(0., 1., n_points)

    grid = np.linspace(0.,1.,n_bins+1)
    sizebin = grid[1]-grid[0]
    plt.figure(figsize=(12, 6))
    for i, a in enumerate(a_collection):
        ax1 = plt.subplot(1, 2, i + 1)

        eta_fun = lambda x: 2*eta_fun_of_a(x, a=a)

        X, Y = datagen_fun_of_a(n=n, a=a)
        gridXpos = grid_count(grid, X[Y == 1])
        gridXneg = grid_count(grid, X[Y ==-1])
        ax2 = ax1.twinx()
        lns1 = ax2.bar(grid[0:n_bins], gridXpos, width=sizebin, align="edge", color="green", alpha=0.5, label="class 1")
        lns2 = ax2.bar(grid[0:n_bins], gridXneg, width=sizebin, align="edge", bottom=gridXpos, color="red", alpha=0.5, label="class 2")
        lns3 = ax1.plot(x_ax, list(map(eta_fun, x_ax)), label="$\mu_1(x)$", color="black", linewidth=3)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="lower right") # "upper left") # 
        if i==0:
            ax1.set_ylabel("$\mu_1(x)$")
        else:
            ax2.set_ylabel("$P_n(X \in $ bin $  \; | \; Y = i)$, $i \in \{1,2\}$" , labelpad=20)
        plt.title("$a = " + str(a)+"$")
        # if i == 1:
        ax1.set_xlabel("$x$")
    plt.tight_layout()
    
def plot_possible_alpha_values(min_max_alpha, a_collection, m, alpha, sup_inc_simrank=False):
    """
    Plot the possible alpha values.
    """
    plt.figure(figsize=(4,3))
    zipped_int_alpha = list(map(lambda a: min_max_alpha(a, m), a_collection))
    inf_alphas = [v[0] for v in zipped_int_alpha]
    sup_alphas = [v[1] for v in zipped_int_alpha]
    plt.plot(a_collection, inf_alphas, label="min for $\\alpha$ ($C>0$)")
    plt.plot(a_collection, sup_alphas, label="max for $\\alpha$ ($C<1/2$)")
    plt.plot(a_collection, [alpha]*len(a_collection), label="chosen $\\alpha$ ")
    if sup_inc_simrank: # Only for the similarity ranking case.
        sup_inc = [0.5 - 0.5*np.power(np.abs(2*m-1), (1-a)/a) for a in a_collection]
        plt.plot(a_collection, sup_inc, label="$\eta$ increasing")
    plt.xlabel("$a$")
    plt.ylabel("$\\alpha$")
    plt.title("Limitations on the possible value for $\\alpha$")
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_emp_mammen(gendata_fun_of_a, eta_fun_of_a, a_collection, n_obs=1000, simrank=False):
    """
    Tries and show visually what the Mammen-Tsybakov assumption means.
    """
    plt.figure(figsize=(3,3))

    for i, a in enumerate(a_collection):
        X, _ = gendata_fun_of_a(n=n_obs, a=a)
        
        if simrank:
            all_pairs = itertools.combinations(X, 2)
            all_vals = list(map(lambda x : abs(eta_fun_of_a(x[0],x[1], a=a)-0.5), all_pairs))
        else:
            all_vals = [np.abs(eta_fun_of_a(x, a=a) - 0.5) for x in X]
        
        plt.hist(all_vals, bins=50, alpha=0.5, cumulative=True,
                 normed=True, label="$a = {:0.2f}$".format(a))

    plt.title("The distribution of $|\eta-1/2|$ for different $a$")
    plt.xlabel("$|\eta-1/2|$")
    plt.ylabel("$P ( X \le t)$")
    plt.legend()
    plt.grid()
    plt.show()