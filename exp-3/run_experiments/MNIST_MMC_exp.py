import autograd.numpy as np
import configargparse
import time

from autograd import grad
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from collections import Counter
from itertools import combinations

def t_print(s,time):
    print(s+".... {:2.2f} s".format(time))

def sq_mahalanobis_dist(A, x, xp):
    """
    Squared Mahalanobis distance d_A^2 : d_A^2(x,xp) = (x-xp)^T A (x-xp).
    Parameters:
    * A: "covariance matrix" of the distance, \in R^{dxd}.
    * x, xp: compared vectors, in R^d.
    Returns:
    * Squared Mahalanobis distance in R.
    """
    delta = x-xp
    return np.dot(delta.transpose(), np.dot(A,delta))

class ExpRecord:
    """
    Class that stores the parameters and results of an experiment.
    """
    def __init__(self, n, status, B, A):
        """
        Parameters:
        * B: Number of tuples used to estimate E [ d_A(X,X') | Y \ne Y'], -1 if complete statistic.
        * status: complete or incomplete experiment.
        * A: solution matrix.
        * n: number of instances.
        """
        self.B = int(B)
        self.n = int(n)
        self.status = status
        self.A = A

    def save_test_obj(self, X, y):
        """
        Computes the test objective and constraint and stores them in the record.
        """
        n_p_pos = 0
        n_p_neg = 0
        self.test_obj = 0
        self.test_cons = 0
        n = X.shape[0]
        for i in range(0,n-1):
            for j in range(i,n):
                dist = sq_mahalanobis_dist(self.A, X[i].reshape((-1,1)), X[j].reshape((-1,1)))
                if y[i] == y[j]:
                    self.test_cons += dist
                    self.n_p_pos += 1
                else:
                    self.test_obj += np.sqrt(dist)
                    self.n_p_neg += 1
                    
        self.test_cons /= self.n_p_pos
        self.test_obj /= self.n_p_neg
                    
    def save_time(self, t):
        self.t = t

    def save_record(self, fold_records):
        """
        Save the record to a file, in folder fold_records.
        """
        if B != -1:
            filename = fold_records+"/exp_B"+str(self.B)+"_n"+str(self.n)+".txt"
        else:
            filename = fold_records+"/exp_comp_n"+str(self.n)+".txt"
        with open(filename, "wt") as f:
            f.write(self.status + " " + str(self.n) + " " + str(self.B) + " " 
                + str(self.test_obj) + "  " + str(self.test_cons) + " " + str(int(self.t)) + "\n")

class MMC:
    """
    See:
        Distance metric learning with application to clustering with side information
        - Xing et al 2002.

    Solves:
        max_A g(A) = \sum_{(x_i,x_j) \in D} || x_i, x_j ||_A
        s.t.  f(A) = \sum_{(x_i,x_j) \in S} || x_i, x_j ||_A^2 <= 1
        with  A > 0
    """
    
    def __init__(self, X, y, iterators, tuple_iter_neg=False, A_0=None, alpha=0.1, epsilon=0.001, 
            verbose=1, init_time=None, steps_print=10):
        """
        About parameters:
        * tuple_iter_neg: if True, we use an incomplete tuple-based statistic to estimate E [ d_A(X,X') | Y \ne Y'].
            If False, we use a complete statistic.
        * A_0: initial value for the "covariance matrix" of the Mahalanobis distance.
        * alpha: step of the gradient descent.
        * epsilon: variation of the objective that make us stop the iterations.
        """ 
        def hermit(x,xp):
            delta = (x-xp).reshape((-1,1))
            return delta.dot(delta.transpose())
        
        self.x_p_constr = np.mean([ hermit(x,xp) for x,xp in iterators["pos"](X,y) ], axis=0)
        
        self.X = X
        self.y = y
        self.it_neg = iterators["neg"]
        self.tuple_iter_neg = tuple_iter_neg
        
        self.tol = 0.01 # tolerance parameter for violating the constraint
        self.eps = np.finfo(float).eps # tolerance parameter for positive eigenvals
        
        # GD parameters
        self.it = 0 # iteration of gradient descent
        self.steps_print = steps_print
        self.epsilon = epsilon
        self.alpha = alpha
        self.obj = - float("inf")
        self.stop_opt = False
        
        if A_0 is None:
            self.A = np.eye(X.shape[1])
        
        self.verbose = verbose
        if init_time is None:
            self.init_time = time.time()
        else:
            self.init_time = init_time
        
        self.project_A()
    
    def objective(self, A):
        sum_maha_dists = 0
        n_elems = 0
        if self.tuple_iter_neg:
            for Xs in self.it_neg(self.X, self.y):
                for x,xp in combinations(Xs, 2):
                    sum_maha_dists += np.sqrt(sq_mahalanobis_dist(A, x, xp))
                    n_elems += 1
            return sum_maha_dists/n_elems
        else:
            for x,xp in self.it_neg(self.X, self.y): 
                sum_maha_dists += np.sqrt(sq_mahalanobis_dist(A, x, xp))
                n_elems += 1
            return sum_maha_dists/n_elems
            
    def project_A(self):
        eigvals, eigvecs = np.linalg.eigh(self.A)
        const_C1 = False
        n_neg_eigvals = (eigvals < 0).sum()
        self.constr = (self.x_p_constr * self.A).sum()
        
        while  n_neg_eigvals > 0 or not  self.constr <= 1 + self.tol:
            if self.verbose > 1:
                t_print("n_neg_eigvals={:2}/constraint={:4.2f}".format(int(n_neg_eigvals), self.constr), 
                    time.time()-self.init_time)                
            self.project_A_on_C1(self.constr) 
            self.project_A_on_C2()
            n_neg_eigvals = 0 # Since we just projected A on C2
            self.constr = (self.x_p_constr * self.A).sum()
    
    def project_A_on_C1(self, constr):
        if constr <= 1:
            self.A = self.A
        else:
            lagrangian_cons = (2*(constr-1))/np.sum(np.power(self.x_p_constr,2))
            self.A = self.A - lagrangian_cons*self.x_p_constr
            
    def project_A_on_C2(self):
        eigvals, eigvecs = np.linalg.eigh(self.A)
        new_eigvals = np.maximum(eigvals,self.eps)
        self.A = eigvecs.dot(np.diag(new_eigvals)).dot(eigvecs.transpose())
        
    def fit(self, n_it=5000):
        grad_obj = grad(self.objective)
        for i in range(0,n_it):
            self.A = self.A + self.alpha*grad_obj(self.A) # gradient ascent
            self.project_A()
            # check for converged objective
            if self.it % self.steps_print == 0:
                new_obj = self.objective(self.A)
                if ( self.epsilon is not None ) and ( np.abs(new_obj - self.obj) < self.epsilon ):
                    break
                self.obj = new_obj
                t_print("it={}/obj={:4.2f}/cons={:4.2f}".format(self.it, self.obj, self.constr), 
                    time.time()-self.init_time)
            self.it +=1

class CompPosIterator:
    """
    Defines an iterator on all of the positive pairs of a sample (X,y).
    """
    def __init__(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_per_class = Counter(y)
        self.d_inds = { k : [] for k in self.classes }
        for i, y_i in enumerate(y):
            self.d_inds[y_i].append(i)
        self.X = X
    
    def __iter__(self):
        self.i = 0
        self.j = 0
        self.Yi = 0
        self.clas_i = self.classes[self.Yi]
        return self
    
    def __next__(self):
        if self.j >= self.n_per_class[self.clas_i]-1:
            if self.i >= self.n_per_class[self.clas_i]-2:
                cond = True
                while cond: # do ... while-type loop
                    if self.Yi == self.n_classes-1:
                        raise StopIteration
                    else:
                        self.Yi += 1
                        self.i = 0
                    self.clas_i = self.classes[self.Yi]
                    cond = ( self.n_per_class[self.clas_i] <= 1 )
            else:
                self.i += 1
            self.j = self.i + 1
        else:
            self.j += 1
        return self.X[self.d_inds[self.clas_i][self.i]], self.X[self.d_inds[self.clas_i][self.j]]


class CompNegIterator:
    """
    Defines an iterator on all of the negative pairs of a sample (X,y).
    """
    def __init__(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_per_class = Counter(y)
        self.d_inds = { k : [] for k in self.classes }
        for i, y_i in enumerate(y):
            self.d_inds[y_i].append(i)
        self.X = X
    
    def __iter__(self):
        self.i = 0
        self.j = 0
        self.Yi = 0
        self.Yj = 1
        self.first_neg = True
        self.clas_i = self.classes[self.Yi]
        self.clas_j = self.classes[self.Yj]
        return self
    
    def __next__(self):
        if not(self.first_neg):
            if self.j == self.n_per_class[self.clas_j]-1:
                if self.Yj == self.n_classes-1:
                    if self.i == self.n_per_class[self.clas_i]-1:
                        if self.Yi == self.n_classes-2:
                            raise StopIteration
                        else:
                            self.Yi += 1
                            self.i = 0                        
                    else:
                        self.i += 1                    
                    self.j = 0
                    self.Yj = self.Yi + 1
                else:
                    self.Yj += 1
                    self.j = 0
                self.clas_i = self.classes[self.Yi]
                self.clas_j = self.classes[self.Yj]
            else:
                self.j += 1
        else:
            self.first_neg = False
        return self.X[self.d_inds[self.clas_i][self.i]], self.X[self.d_inds[self.clas_j][self.j]]
        
class IncompNegIterator:
    """
    Defines an incomplete iterator based on B tuples for the negative pairs of a sample (X,y).
    """
    def __init__(self, X, y, B):
        self.classes = np.unique(y)
        self.d_inds = { k : [] for k in self.classes }
        for i, y_i in enumerate(y):
            self.d_inds[y_i].append(i)
        self.B = B
        self.X = X
    
    def __iter__(self):
        self.i_pair = 0
        return self
    
    def __next__(self):
        if self.i_pair == self.B:
            raise StopIteration
        self.i_pair += 1
        return [ self.X[np.random.choice(self.d_inds[k])] for k in self.classes]


if __name__ == '__main__':
    init_time = time.time()
    t_print("Starting",time.time() - init_time)

    # --- Parsing the arguments of program ---
    parser = configargparse.ArgumentParser()
    # - Required -
    parser.add_argument("--n_choice", type=int, help="n", required=True)
    parser.add_argument("--type_exp", type=str, help="complete or incomplete", required=True)
    # - Non required -
    parser.add_argument("--variancePCA", type=float, help="Variance kept after PCA", default=0.90)
    parser.add_argument("--testsize", type=float, help="Test size", default=1/7)
    parser.add_argument("--ratio", type=float, help="Ratio of B/n")
    parser.add_argument("--fold_record", type=str, help="Outside folder", 
        default="records/")
    args = parser.parse_args()
    
    variance_percentage = args.variancePCA
    test_size = args.testsize
    ratio = args.ratio
    n_choice = args.n_choice
    type_exp = args.type_exp
    fold_record = args.fold_record
    
    if args.ratio is None and args.type_exp == "incomplete":
        raise ValueError('You did not supply a ratio for B/N while asking for incomplete U-stats experiments...')

    # --- Loading of the data ---
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    y = mnist.target
    
    n_classes = np.unique(y).shape[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # --- Transformation of the data by PCA learned on train set ---
    n_components = 100

    pca = PCA(n_components=n_components, whiten=True).fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    t_print("Done PCA", time.time() - init_time)
    
    n_kept_axes = np.argmax(pca.explained_variance_ratio_.cumsum() > variance_percentage)

    X_train_pca_red = X_train_pca[:, :n_kept_axes]
    X_test_pca_red = X_test_pca[:, :n_kept_axes]
    
    # --- We keep only the first n_choice elements to learn on ---
    X_train_pca_red = X_train_pca_red[:n_choice]
    y_train = y_train[:n_choice]


    # --- We derive the iterators based on the type of experiment ---
    if type_exp == "complete":
        B = -1
        iterators = {"pos" : CompPosIterator, "neg" : CompNegIterator} 
        tuple_iter_neg = False
    else:
        B = int(ratio*n_choice)
        selected_elems = IncompNegIterator(range(0,len(y_train)), y_train, B)
        iterators = {"pos" : CompPosIterator, "neg" : lambda X,y : ( [ X[i] for i in i_s ] for i_s in selected_elems ) }
        tuple_iter_neg = True
    
    # --- Run the experiment and write it to a record ---
    t_print("Experiment n= {} / B = {}".format(n_choice, B), time.time() - init_time)
    begin_time = time.time()
    
    mmc = MMC(X_train_pca_red, y_train, iterators, tuple_iter_neg=tuple_iter_neg, init_time=init_time)
    mmc.fit(n_it=5000)
        
    record = ExpRecord(B, type_exp, mmc_prob.A, n_choice)
    record.save_time(time.time() - begin_time)
    record.get_test_res(X_test_pca_red, y_test)
    
    t_print("Test obj = {:4.2f}".format(record.test_obj),time.time()-init_time)
    record.save_record(fold_record)
