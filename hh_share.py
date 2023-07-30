import numpy as np
import pdb
import sympy
from functools import partial
from copy import deepcopy
import qbee


def hodgkin_huxley_sympy():

    # Define the model variables
    V, m, h, n, I = sp.symbols('V m h n I')

    # Define the model parameters
    C_m = 1.0  # Membrane capacitance (uF/cm^2)
    g_Na = 120.0  # Maximum sodium conductance (mS/cm^2)
    g_K = 36.0  # Maximum potassium conductance (mS/cm^2)
    g_L = 0.3  # Leak conductance (mS/cm^2)
    E_Na = 50.0  # Sodium reversal potential (mV)
    E_K = -77.0  # Potassium reversal potential (mV)
    E_L = -54.387  # Leak reversal potential (mV)

    # Define the gating variable equations
    alpha_m = 0.1 * (V + 40) / (1 - sp.exp(-0.1 * (V + 40)))
    beta_m = 4 * sp.exp(-0.0556 * (V + 65))
    alpha_h = 0.07 * sp.exp(-0.05 * (V + 65))
    beta_h = 1 / (1 + sp.exp(-0.1 * (V + 35)))
    alpha_n = 0.01 * (V + 55) / (1 - sp.exp(-0.1 * (V + 55)))
    beta_n = 0.125 * sp.exp(-0.0125 * (V + 65))

    # Define the membrane current equations
    # Do not include external current
    I_Na = g_Na * m ** 3 * h * (V - E_Na)
    I_K = g_K * n ** 4 * (V - E_K)
    I_L = g_L * (V - E_L)

    # Define the membrane potential equation
    # NOTE: We have neglected output currents here
    dVdt = (I_Na - I_K - I_L) / C_m

    # Define the gating variable equations
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    dndt = alpha_n * (1 - n) - beta_n * n

    # return in a format that can be processed by the polynomialization algorithm
    return [[V, dVdt], [m, dmdt], [h, dhdt], [n, dndt]]



###### Custom polynomializaton code #####

def traverse_nonpolynomials(expr, depth=0):
    """
    Recursively traverses a SymPy expression and returns a list of its maximal non-polynomial subterms.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to traverse.

    Returns
    -------
    list
        A list of the maximal non-polynomial subterms of the expression.

    Examples
    --------
    >>> from sympy.abc import x, y, z
    >>> from sympy import sin, cos
    >>> traverse_nonpolynomials(x**2 + y*sin(x) + z*cos(y))
    [sin(x), cos(y)]
    """

    terms = []
    if expr.is_polynomial():
        return terms
    else:
        for arg in expr.args:
            if arg.is_polynomial():
                continue
            else:
                subterms = traverse_nonpolynomials(arg, depth=depth+1)
                if subterms:
                    subterms.insert(0, arg)
                    if depth == 0:
                        terms.append(subterms)
                    else:
                        terms.extend(subterms)
                else:
                    terms.append(arg)
    return terms

def depth_until_polynomial(expr, depth=0):
    """
    Computes the depth of a SymPy expression until a polynomial is reached.
    
    Parameters
    ----------
    expr : sympy.Expr
        The expression to compute the depth of.

    Returns
    -------
    int
        The depth of the expression.

    Examples
    --------
    >>> from sympy.abc import x, y, z
    >>> from sympy import sin, cos
    >>> polynomial_depth(x**2 + y*sin(x) + z*cos(y))
    1
    """

    if expr.is_polynomial():
        return depth
    else:
        depth += 1
        child_depths = []
        for arg in expr.args:
            child_depths.append(depth_until_polynomial(arg, 0))
        return depth + max(child_depths)

# Apply sympy substitution only if the expression is not already a poilynomial
# and applying the transformation would reduce the depth until one obtains a polynomial
def conditional_substitution(expr, substitutions):
    if len(substitutions) == 0:
        return expr
    else:
        # Separate sums of terms
        if expr.func == sympy.Add or expr.func == sympy.Mul:
            args = []
            for arg in expr.args:
                args.append(conditional_substitution(arg, substitutions))
            return expr.func(*args)        
        elif not expr.is_polynomial():
            # At each step, choose the substitution that reduces the depth the most
            depth_reduc = []
            for key, value in substitutions.items():
                depth_reduc.append(depth_until_polynomial(expr) - depth_until_polynomial(expr.subs(key, value)))
            if max(depth_reduc) > 0:
                subs = list(substitutions.items())
                expr = expr.subs(*subs[np.argmax(depth_reduc)])
                remaining_substitutions = deepcopy(substitutions)
                del remaining_substitutions[subs[np.argmax(depth_reduc)][0]]
                return conditional_substitution(expr, remaining_substitutions)
            else:
                return expr
        else:
            return expr

def polynomialize(expressions):

    # For each variable, create a dot symbol to represent its time derivative. This is employed in the Lie derivatives, 
    # and then substituted at the very end
    dots = {}
    for expression in expressions:
        vdot = sympy.symbols(str(expression[0]) + 'dot')
        dots[expression[0]] = vdot

    odots = deepcopy(dots)

    symidx = 0
    transformations = {}
    # Loop through expression list until all expressions are polynomials
    all_poly = False
    iter_ = 0
    while not all_poly:
        all_poly = True
        n_poly = 0
        print(n_poly)
        print(len(expressions))
        for expression in expressions:
            # First apply existing transformations to expression
            expr = conditional_substitution(expression[1], transformations)

            # Not sure how robust this check is
            if expr.is_polynomial():
                expression[1] = expr
                n_poly += 1
            else:
                all_poly = False
                # Recursively traverse the expression to find all non polynomial subexpressions
                terms = traverse_nonpolynomials(expr)
                # Squash nested lists to depth 1

                for term in terms:
                    
                    # If term is a composition of elementary functions, need to traverse the list in reverse order
                    if isinstance(term, list):
                        # Need to take derivatives with respect to subsequent nested variables, and then substitute the transformations
                        nested_vars = []
                        nested_derivatives = []

                        for idx, term_ in enumerate(reversed(term)):
                            # If after applying the transformation, the term is a polynomial, then we can stop
                            try:
                                term_subs = conditional_substitution(term_, transformations)
                            except:
                                pdb.set_trace()
                            if term_subs.is_polynomial():
                                break
                            var_ = sympy.symbols('y' + str(symidx))
                            symidx += 1
                            nested_vars.append(var_)
                            transformations[term_subs] = var_
                            if idx == 0:
                                # Use of free symbols is done to ignore constants in args
                                Ld = sympy.diff(term_, term_.args[0].free_symbols.pop()) * dots[expression[0]]
                                nested_derivatives.append(conditional_substitution(Ld, transformations))
                            else:
                                Ld = sympy.diff(term_subs, nested_vars[idx-1]) * nested_derivatives[idx - 1]
                                nested_derivatives.append(conditional_substitution(Ld, transformations))
                            # Append to expressions
                            dots[var_] = conditional_substitution(Ld, transformations)
                    else:    
                        # Let y1 = g(x)
                        var_ = sympy.symbols('y' + str(symidx))
                        symidx += 1
                        # Add the transformation to the dictionary
                        transformations[term] = var_
                        # Lie derivative

                        # Use of free symbols is done to ignore constants in args
                        Ld = sympy.diff(term, term.args[0].free_symbols.pop()) * dots[expression[0]]

                        # Substitute existing transformations
                        Ld = conditional_substitution(Ld, transformations)
                        # Append to expressions
                        dots[var_] = Ld
                
            expression[1] = conditional_substitution(expression[1], transformations)    

        # Simplify dots via substitutions
        for key, value in dots.items():
            dots[key] = conditional_substitution(value, transformations)
        
        # Add to expressions the new ODEs contained in dots
        new_expressions = []
        for key, value in dots.items():
            if key not in [e[0] for e in expressions]:
                new_expressions.append([key, value])
        expressions += new_expressions
        #print(expressions)
        #print(dots)
        #expressions += new_expressions
        #print(expressions)
        #print(transformations)
        # iter_ += 1
        # if iter_ >= 2:
        #     break

    # Original dots have been polynomialized, so substitute them back in
    for key, value in odots.items():
        for expression in expressions:
            expression[1] = expression[1].subs(value, [e for e in expressions if e[0] == key][0][1])

    # Convert to expressions that can be numerically evaluated
    # for expression in expressions:
    #     expression[1] = sympy.lambdify([e[0] for e in expressions], expression[1])

    return expressions, transformations

# Convert expressions to a function compatible with solve_ivp
def numericalize(expressions):
    num_expressions = [sympy.lambdify([e[0] for e in expressions], exp[1], modules='scipy') for exp in expressions]

    def num_func(t, x):
        return np.array([e(*x) for e in num_expressions])
    return num_func

# Convert our format to something suitable for qbee
def qbee_preprocess(expr):

    qbee_expr = []
    vars = [str(e[0]) for e in expr]
    f = qbee.functions(', '.join(vars))

    # Substitute sympy symbols for qbee functions 

    for i, e in enumerate(expr):
        qbee_expr.append((f[i], e[1].subs({v:f[j] for j, v in enumerate(vars)})))

    return qbee_expr

if __name__ == '__main__':

    expr = hodgkin_huxley_sympy()
    
    # Using custom polynomialization code, not qbee
    model, trans = polynomialize(expr)

    qbeqs = qbee_preprocess(model)
    quad_upper_bound = partial(qbee.pruning_by_vars_number, nvars=31)  # We need some upper bound
    I = qbee.functions('I')
    quadobj = qbee.quadratize(qbeqs, pruning_functions=[quad_upper_bound], calc_upper_bound=False, input_free=False, input_der_orders={I:2})