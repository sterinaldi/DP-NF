import numpy as np
from scipy.special import logsumexp
from figaro.mixture import DPGMM, density, _update_alpha
# FIXME: Import your favourite normalising flow

class _component:
    """
    Class to store a normalising flow.
    
    Arguments:
        np.ndarray x: sample added to the new component
        prior prior:  instance of the prior class with NIG/NIW prior parameters
    
    Returns:
        component: instance of component class
    """
    def __init__(self, x, prior, N = 1):
        self.N     = N
        self.prior = prior
        # FIXME: instantiate a new flow
        self.flow  = my_favourite_flow()
    
    def __call__(self, x):
        return self.pdf(x)
    
    def pdf(self, x):
        # FIXME: is this how you evaluate a flow?
        return self.flow.pdf(x)
    
    def logpdf(self, x):
        # FIXME: is this how you evaluate a flow?
        return self.flow.logpdf(x)
    
    def train(self, x):
        # FIXME: I have no idea
        pass
    
class mixture(density):
    """
    Class to store a single draw from DPGMM/(H)DPGMM.
    Methods inherited from density class
    
    Arguments:
        iterable components: NF components
        np.ndarray w:        component weights
        np.ndarray bounds:   bounds of probit transformation
        int dim:             number of dimensions
        int n_cl:            number of clusters in the mixture
        int n_pts:           number of points used to infer the mixture
        double alpha:        concentration parameter
        bool probit:         whether to use the probit transformation or not
        np.ndarray log_w:    component log weights
        bool make_comp:      make component objects
        double alpha_factor: evaluated \\int pdet(theta)p(theta|lambda) conditioned on the mixture parameters
    
    Returns:
        mixture: instance of mixture class
    """
    def __init__(self, components,
                       w,
                       bounds,
                       dim,
                       n_cl,
                       n_pts,
                       alpha        = 1.,
                       probit       = True,
                       log_w        = None,
                       alpha_factor = 1.,
                       ):
        self.components   = components
        if log_w is None:
            self.w        = w
            self.log_w    = np.log(w)
        else:
            self.log_w    = log_w
            self.w        = np.exp(log_w)
        self.bounds       = bounds
        self.dim          = int(dim)
        self.n_cl         = int(n_cl)
        self.n_pts        = int(n_pts)
        self.probit       = probit
        self.alpha        = alpha
        self.alpha_factor = alpha_factor
    
    def _pdf_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return np.sum(np.array([w*flow.pdf(x) for flow, w in zip(self.components, self.w)]), axis = 0)
    
    def _logpdf_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return logsumexp(np.array([w + flow.logpdf(x) for flow, w in zip(self.components, self.log_w)]), axis = 0)
    
    def _rvs_probit(self, size = 1):
        """
        Draw samples from mixture in probit space
        
        Arguments:
            int size: number of samples to draw
        
        Returns:
            np.ndarray: samples in probit space
        """
        idx = np.random.choice(np.arange(self.n_cl), p = self.w, size = size)
        ctr = Counter(idx)
        if self.dim > 1:
            samples = np.empty(shape = (1,self.dim))
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(self.components[i].rvs(size = n))))
        else:
            samples = np.array([np.zeros(1)])
            for i, n in zip(ctr.keys(), ctr.values()):
                samples = np.concatenate((samples, np.atleast_2d(self.components[i].rvs(size = n)).T))
        return np.array(samples[1:])

class DPNF(DPGMM):
    """
    Dirichlet Process Normalising Flow Mixture Model class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Can't un-train a flow (can we?)
        self.n_reassignments = 0
    
    def _add_datapoint_to_component(self, x, component):
        """
        This method trains the selected flow using the sample x
        
        Arguments:
            np.ndarray x:         sample
            _component component: NF to train
        
        Returns:
            component: trained NF
        """
        component.train(x)
        component.N += 1
        return component
    
    def _log_predictive_likelihood(self, x, component):
        """
        Compute log likelihood of drawing sample x from a specific flow given the samples on which the flow is already trained on
        
        Arguments:
            np.ndarray x: sample
            component component: NF to evaluate
        
        Returns:
            double: log Likelihood
        """
        if component is None:
            # FIXME: New component, from the prior (John, you mentioned something?)
        return component.logpdf(x)

    def build_mixture(self, make_comp = True):
        """
        Instances a mixture class representing the inferred distribution
        
        Arguments:
            bool make_comp: compatibility with FIGARO
        
        Returns:
            mixture: the inferred distribution
        """
        if self.n_cl == 0:
            raise FIGAROException("You are trying to build an empty mixture - perhaps you called the initialise() method. If you are using the density_from_samples() method, the inferred mixture is returned by that method as an instance of mixture class.")
        # Number of active clusters
        n_cl = (np.array(self.N_list) > 0).sum()
        for ss in self.mixture:
            if ss.N > 0:
                components.append(ss.flow)
        w = dirichlet(self.w[self.w > 0]*self.n_pts+self.alpha/self.n_cl).rvs()[0]
        return mixture(components,
                       w,
                       self.bounds,
                       self.dim,
                       n_cl,
                       self.n_pts,
                       self.alpha,
                       probit = self.probit,
                       )
