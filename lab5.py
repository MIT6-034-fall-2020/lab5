# MIT 6.034 Lab 5: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Returns a set containing the ancestors of var"
    # use DFS queue approach to work backwards
    ancestors = net.get_parents(var)
    queue = list(net.get_parents(var))

    while queue:
        curr = queue[0]
        queue.pop(0)
        queue.extend(list(net.get_parents(curr)))
        ancestors.update(net.get_parents(curr))
    
    return ancestors

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    # use DFS queue approach to work forward
    descendants = net.get_children(var)
    queue = list(net.get_children(var))

    while queue:
        curr = queue[0]
        queue.pop(0)
        queue.extend(list(net.get_children(curr)))
        descendants.update(net.get_children(curr))
    
    return descendants

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    # Set operation: nondescedants = all nodes - descedeants - self 
    all_vars = set(net.get_variables())
    descendants = get_descendants(net, var)
    self_set = {var}
    nondescendants = all_vars - descendants - self_set
    return nondescendants



#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    raise NotImplementedError
    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    raise NotImplementedError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    raise NotImplementedError
    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    raise NotImplementedError

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    raise NotImplementedError
    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    raise NotImplementedError


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    raise NotImplementedError


#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    raise NotImplementedError
    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    raise NotImplementedError


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
