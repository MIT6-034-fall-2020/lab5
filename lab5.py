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
    
    Algorithm:
    - Get descedants of var, check if overlap with given (intersection not empty)
        - If true: return given 
        - If false, move to the next check
    - Get the parents and check if subset of given
        - If true, remove all nondescendants except parents (i.e. return parents)
        - If false, return given
    """

    # make set of conditional variables
    given_vars = set(givens.keys())

    # checks for descendants in the given (true if empty)
    descendants = get_descendants(net, var)
    if given_vars.intersection(descendants): 
        return givens

    # check for parents in given
    parents = net.get_parents(var)
    if parents.issubset(given_vars):
        # new_givens = {}
        # for parent in parents:
        #     new_givens[parent] = givens[parent]
        return {p: givens[p] for p in parents}
    else:
        return givens

    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    """
    Algorithm:
    - hypothesis should be singleton, givens should be parents or simplify to them
    - check hypothesis size for length 1
    - two cases: 
        1. given = None 
            check the hypothesis (do error checkup)
                with error, only return LookupError
        2. given != None
            simplify_givens
            check hypothesis (do error checkup)
    """
    # checks if hypothesis has one var
    if len(hypothesis.keys()) != 1:
        raise LookupError
    else:
        var = list(hypothesis.keys())[0]

    if givens == None:
        try:
            return net.get_probability(hypothesis)
        except:
            raise LookupError
    else:
        simplified_givens = simplify_givens(net, var, givens)
        try:
            return net.get_probability(hypothesis, simplified_givens)
        except:
            raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    """
    Hypothesis represents a valid joint probability (that is, contains every variable in the Bayes net).
    Utilize topological sort (from bottom to top) to find order and then chain rule
    """
    varlist = net.topological_sort()
    varlist.reverse()
    probs = []
    for i, var in enumerate(varlist):
        prob = probability_lookup(net, {var: hypothesis[var]}, {v: hypothesis[v] for v in varlist[i+1:]})
        probs.append(prob)
    return product(probs)
    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    """
    Find all the combinations of joints where the hypothesis conditional holds
    Then sum over to get the marginal
    """
    all_vars = net.get_variables()
    joints = net.combinations(all_vars, hypothesis)
    prob = sum([probability_joint(net, j) for j in joints])
    return prob

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
