import copy
import math
import signal
import warnings
import sympy as sp
from sympy.core.function import AppliedUndef
from typing import Iterator
from ordered_set import OrderedSet
from .util import *
from .printer import str_qbee


class Parameter(sp.Symbol):
    pass


class VariablesHolder:
    """
    Class that manages variable storage.
    """

    def __init__(self, variables: Iterable[sp.Symbol],
                 parameter_variables: set[sp.Symbol] | None = None,
                 input_variables: set[sp.Symbol] | None = None,
                 new_var_base_name="w_",
                 start_new_vars_with=0):
        if parameter_variables is None:
            parameter_variables = set()
        if input_variables is None:
            input_variables = set()

        self._parameter_variables = parameter_variables
        self._input_variables = input_variables
        self._state_variables = list(variables)
        self._generated_variables = list()
        self._base_name = new_var_base_name
        self._start_id = start_new_vars_with

    @property
    def state(self):
        # State variables are in a list to preserve ordering introduced in equation systems
        return self._state_variables

    @property
    def parameter(self):
        return self._parameter_variables

    @property
    def input(self):
        return self._input_variables

    @property
    def generated(self):
        return self._generated_variables

    @property
    def base_var_name(self):
        return self._base_name

    @base_var_name.setter
    def base_var_name(self, value: str):
        self._base_name = value

    @property
    def start_new_vars_with(self):
        return self._start_id

    @start_new_vars_with.setter
    def start_new_vars_with(self, value):
        self._start_id = value

    def create(self) -> sp.Symbol:
        """
        Creates a new variable and stores it within itself.

        :Example:

        >>> self.create()
        w1
        >>> self.create()
        w2
        >>> self.create()
        w3

        :return: Created variable

        """
        new_index = len(self._generated_variables) + self._start_id
        new_variable = sp.Symbol(self._base_name + "{%d}" % new_index)

        self._state_variables.append(new_variable)
        self._generated_variables.append(new_variable)
        return new_variable


class EquationSystem:
    def __init__(self, equations: dict[sp.Symbol, sp.Expr],
                 parameter_variables: Iterable[sp.Symbol] = None,
                 input_variables: Iterable[sp.Symbol] = None):
        self._equations = equations.copy()
        self._substitution_equations: dict[sp.Symbol, sp.Expr] = dict()
        self._poly_equations: dict[sp.Symbol, sp.Expr | None] = {k: None for k in equations.keys()}

        _parameter_vars = set(parameter_variables) if parameter_variables is not None else set()
        _input_vars = set(input_variables) if input_variables is not None else set()
        _variables = list(equations.keys())
        self.variables = VariablesHolder(_variables, _parameter_vars, _input_vars)

        self.expand()
        self._fill_poly_system()

    def __copy__(self):
        system = EquationSystem(self._equations)
        system._equations = {k: v for k, v in self._equations.items()}
        system._substitution_equations = {k: v for k, v in self._substitution_equations.items()}
        system._poly_equations = {k: v for k, v in self._poly_equations.items()}
        system.variables = pickle.loads(pickle.dumps(self.variables, -1))  # fast deepcopy
        return system

    @property
    def equations(self) -> list[sp.Eq]:
        return [sp.Eq(make_derivative_symbol(dx), f) for dx, f in self._equations.items()]

    @property
    def substitution_equations(self):
        return [sp.Eq(x, f) for x, f in self._substitution_equations.items()]

    @property
    def polynomial_equations(self):
        return [sp.Eq(make_derivative_symbol(x), f) for x, f in self._poly_equations.items()]

    def to_poly_equations(self, inputs_ord: dict):
        """System should be already polynomial. Otherwise, throws Exception. You can check it by `is_polynomial`
        method """
        # TODO: Does not work correctly. Try checking self.variables.state
        assert self.is_polynomial()
        inputs_ord_sym = {sp.Symbol(str_qbee(k)): v for k, v in inputs_ord.items()}
        d_inputs = generate_derivatives(inputs_ord_sym)
        # TODO: Make explicit names for the highest order derivatives instead of 0
        coef_field = sp.FractionField(sp.QQ, list(map(str, self.variables.parameter)))
        R = coef_field[list(self.variables.state + sp.flatten(d_inputs))]
        equations = [R.from_sympy(eq.rhs) for eq in self.polynomial_equations]
        for i, v in enumerate(inputs_ord_sym.keys()):
            for dv in [g for g in R.gens if str(v) + '\'' in str(g)]:
                equations.append(dv)
            equations.append(R.zero)
        inputs_to_exclude = [tuple(R.from_sympy(v[-1])) for v in d_inputs]
        return equations, inputs_to_exclude

    def expand(self):
        """Apply SymPy 'expand' function to each of equation."""
        for x, fx in self._equations.items():
            self._equations[x] = fx.expand()

    def is_polynomial(self) -> bool:
        return all(self._poly_equations.values())

    def add_new_var(self, new_var: sp.Symbol, substitution: sp.Expr) -> None:
        """
        Add a `new_var = substitution` to the system.

        Effects:
        * can add more than one variable recursively
        * can change whether the system is polynomial or not
        """
        if substitution.is_Pow and (
                substitution.exp.is_Float or (substitution.exp < 0 and substitution.exp != -1)) and (
                1 / substitution.base) not in self._substitution_equations.values():
            self.add_new_var(new_var, 1 / substitution.base)
            new_var = self.variables.create()
        self._substitution_equations[new_var] = substitution
        self._equations[new_var] = self._calculate_Lie_derivative(substitution)

        self._fill_poly_system()

    def _calculate_Lie_derivative(self, expr: sp.Expr) -> sp.Expr:
        """Calculates Lie derivative using chain rule."""
        result = sp.Integer(0)
        for var in expr.free_symbols.difference(self.variables.parameter).difference(self.variables.input):
            var_diff = self._equations[var]
            if isinstance(expr, sp.Pow) and isinstance(expr.exp, sp.Float):
                result += sp.Mul(expr.exp * var_diff, expr, sp.Pow(var, -1), evaluate=False)
            elif isinstance(expr, sp.Pow) and expr.exp == sp.Integer(-1):
                result += sp.Mul(var_diff, expr.diff(var), evaluate=False)
            else:
                result += expr.diff(var) * var_diff
        for input_var in expr.free_symbols.intersection(self.variables.input):
            input_var_dot = make_derivative_symbol(input_var)
            self.variables.input.add(input_var_dot)
            result += expr.diff(input_var) * input_var_dot
        return result

    def _fill_poly_system(self):
        """Fills `self._poly_equations` with equations that can be rewritten as polynomial"""
        for x, fx in self._equations.items():
            if x not in self._poly_equations.keys():
                self._poly_equations[x] = None
            if not self._poly_equations[x]:
                self._poly_equations[x] = self._try_convert_to_polynomial(fx)

    def _try_convert_to_polynomial(self, expr: sp.Expr) -> sp.Expr | None:
        # It could be worthy to check every possible permutation, with `subs` especially.
        # subs_permutations = permutations(self._substitution_equations)

        # The order of `replaced_non_pow` and `replaced_pow` is critical
        # since we substitute functions without new variables:
        # sin(1/x) != sin(w)

        try:
            with sp.evaluate(False):
                replaced_non_pow_lazy = expr.subs(
                    {v: k for k, v in self._substitution_equations.items() if not v.is_Pow})
            replaced_pow_lazy = replaced_non_pow_lazy \
                .replace(sp.Pow, self._replace_irrational_pow) \
                .replace(sp.Pow, self._replace_negative_integer_pow)
        except Exception as e:
            warnings.warn("Substituting new variables failed to produce the expected calculations,"
                          " so the calculation was done in an alternative mode. "
                          "If you see this message, please let us know in the Issues: "
                          "https://github.com/AndreyBychkov/QBee/issues", RuntimeWarning)
            replaced_pow_lazy = None

        lazy_res = replaced_pow_lazy if replaced_pow_lazy and self._is_expr_polynomial(replaced_pow_lazy) else None
        if lazy_res:
            return lazy_res

        # eager evaluation
        replaced_non_pow = expr.subs({v: k for k, v in self._substitution_equations.items() if not v.is_Pow})
        replaced_pow = replaced_non_pow \
            .replace(sp.Pow, self._replace_irrational_pow) \
            .replace(sp.Pow, self._replace_negative_integer_pow)
        return replaced_pow if self._is_expr_polynomial(replaced_pow) else None

    def _replace_negative_integer_pow(self, base, exp):
        new_var = key_from_value(self._substitution_equations, (1 / base).subs(self._substitution_equations))
        if exp.is_Integer and new_var:
            return new_var ** (-exp)
        return base ** exp

    def _replace_irrational_pow(self, base, exp):
        new_var = key_from_value(self._substitution_equations, base ** exp)
        if exp.is_Float and new_var:
            return new_var
        return base ** exp

    def _is_expr_polynomial(self, expr: sp.Expr):
        return expr.is_polynomial(*self.variables.state, *self.variables.input)

    def print(self, str_func=str_qbee, use_poly_equations=True):
        """
        Prints equations

        :param str_func: function that stringify Sympy objects.
         Refer to https://docs.sympy.org/latest/tutorials/intro-tutorial/printing.html
        :param use_poly_equations: if True prints polynomial equations. Otherwise, prints equations without substitutions
        """

        equations = self.polynomial_equations if use_poly_equations else self.equations
        print("\n".join(map(str_func, equations)))

    def substitution_equations_str(self):
        return '\n'.join(map(str_qbee, self.substitution_equations))

    def print_substitutions(self, str_func=str_qbee):
        print('\n'.join(map(str_func, self.substitution_equations)))

    def __str__(self):
        equations = self.polynomial_equations if self.is_polynomial() else self.equations
        return '\n'.join(map(str_qbee, equations))

    def __len__(self):
        return len(self._equations)


ALGORITHM_INTERRUPTED = False


def signal_handler(sig_num, frame):
    global ALGORITHM_INTERRUPTED
    print("The algorithm has been interrupted. Returning the current best.")
    ALGORITHM_INTERRUPTED = True


signal.signal(signal.SIGINT, signal_handler)


def polynomialize(system: EquationSystem | list[(sp.Symbol, sp.Expr)], upper_bound=10, new_var_name="w_",
                  start_new_vars_with=0) -> EquationSystem:
    """
    Transforms the system into polynomial form using variable substitution techniques.

    :param system: non-linear ODEs system
    :param upper_bound: a maximum number of new variables that algorithm can introduce.
     If infinite, the algorithm may never stop.
    :param new_var_name: base name for new variables. Example: new_var_name='w' => w0, w1, w2, ...
    :param start_new_vars_with: Initial index for new variables. Example: start_new_vars_with=3 => w3, w4, ...

    """
    if not isinstance(system, EquationSystem):
        system = eq_list_to_eq_system(system)
    system.variables.base_var_name = new_var_name
    system.variables.start_new_vars_with = start_new_vars_with
    nvars, opt_system, traversed = poly_algo_step(system, upper_bound, math.inf)
    return opt_system


def eq_list_to_eq_system(system: List[Tuple[sp.Symbol, sp.Expr]]) -> EquationSystem:
    lhs, rhs = zip(*system)
    params = set(reduce(lambda l, r: l | r, [eq.atoms(Parameter) for eq in rhs]))
    funcs = set(reduce(lambda l, r: l | r, [eq.atoms(AppliedUndef) for eq in rhs]))
    lhs_args = set(sp.flatten([eq.args for eq in lhs if not isinstance(eq, sp.Derivative)]))
    inputs = set(filter(lambda f: (f not in lhs) and (f not in lhs_args), funcs))
    spatial = funcs.difference(inputs)

    degrade_to_symbol = {s: sp.Symbol(str_qbee(s)) for s in funcs | inputs | params | set(lhs)}

    params_sym = [p.subs(degrade_to_symbol) for p in params]
    funcs_sym = [f.subs(degrade_to_symbol) for f in funcs]
    inputs_sym = [i.subs(degrade_to_symbol) for i in inputs]
    spacial_sym = [s.subs(degrade_to_symbol) for s in spatial]

    lhs_sym = [s.subs(degrade_to_symbol) for s in lhs]
    rhs_sym = [eq.subs(degrade_to_symbol) for eq in rhs]

    der_inputs = set(reduce(lambda l, r: l | r, [eq.atoms(sp.Derivative) for eq in rhs]))
    degrade_to_symbol.update({i: sp.Symbol(str_qbee(i)) for i in der_inputs})
    rhs_sym = [eq.subs(degrade_to_symbol) for eq in rhs]
    system = EquationSystem({dx: fx for dx, fx in zip(lhs_sym, rhs_sym)}, params_sym, inputs_sym)
    system.variables._state_variables = list(OrderedSet(system.variables.state + spacial_sym))
    return system


@progress_bar(is_stop=False)
def poly_algo_step(part_res: EquationSystem, upper_bound, best_nvars) -> (int, EquationSystem | None, int):
    """
    Recursive step of Branch and Bound algorithm. Tries to add new variables to `part_res`
    and chooses a system with the smallest number of new variables among polynomial ones.

    :param part_res: current system;
    :param upper_bound: maximum number of new variables, use it to prevent infinite recursion;
    :param best_nvars: current best number of new variables;

    :returns: current best number of variables, current system, and how many nodes traversed the current step;
    """
    if part_res.is_polynomial():
        return len(part_res.variables.generated), part_res, 1
    if len(part_res.variables.generated) >= best_nvars - 1 or \
            len(part_res.variables.generated) >= upper_bound or ALGORITHM_INTERRUPTED:
        return math.inf, None, 1

    traversed_total = 1
    min_nvars, best_system = best_nvars, None
    for next_system in next_gen(part_res):
        nvars, opt_system, traversed = poly_algo_step(next_system, upper_bound, min_nvars)
        traversed_total += traversed
        if nvars < min_nvars:
            min_nvars = nvars
            best_system = opt_system
    return min_nvars, best_system, traversed_total


@progress_bar(is_stop=True)
def final_iter():
    pass


def next_gen(system: EquationSystem) -> Iterator[EquationSystem]:
    return (apply_substitution(system, subs) for subs in available_substitutions(system))


def apply_substitution(system: EquationSystem, subs: sp.Expr) -> EquationSystem:
    # new_system: EquationSystem = pickle.loads(pickle.dumps(system, -1))  # fast deepcopy
    new_system = copy.copy(system)
    new_var = new_system.variables.create()
    new_system.add_new_var(new_var, subs)
    return new_system


def available_substitutions(system: EquationSystem) -> set[sp.Expr]:
    non_poly_equations = [eq for eq, peq in zip(system.equations, system.polynomial_equations) if not peq.rhs]
    subs = [find_nonpolynomial_terms(eq.rhs, set(system.variables.state) | system.variables.input)
            for eq in non_poly_equations]
    non_empty_subs = filter(lambda s: s, subs)
    unused_subs = set(filter(lambda s: s not in system._substitution_equations.values(), sp.flatten(non_empty_subs)))
    return unused_subs


def find_nonpolynomial_terms(expr: sp.Expr, variables) -> set:
    return expr.find(lambda subexpr: is_nonpolynomial_function(subexpr, variables))


def is_nonpolynomial_function(expr: sp.Expr, variables) -> bool:
    negative_cond = [expr.is_polynomial(*variables), expr.is_Add, expr.is_Mul]
    return not any(negative_cond)
