[![Build Status](https://travis-ci.com/AndreyBychkov/QBee.svg?branch=dev)](https://app.travis-ci.com/github/AndreyBychkov/QBee)

# QBee

Python library for transforming systems of ODE equations into a systems with quadratic right-rand side.

# Installation

1. Clone repository: `https://github.com/AndreyBychkov/QBee.git`
2. Install requirements: `pip install -r requirements.txt`
3. Install the package: `pip install .`

# What is quadratization?

The problem of *quadratization* is, given a system of ODEs with polynomial right-hand side, reduce the system to a
system with quadratic right-hand side by introducing as few new variables as possible. We will explain it using toy
example. Consider the system

<img alt="\begin{cases} x_1&#39; = x_1 x_2 \\ x_2&#39; = -x_1 x_2^3 \end{cases}" height="135" src="https://latex.codecogs.com/png.latex?\dpi{200}&amp;space;\huge&amp;space;{\color{DarkOrange}&amp;space;\begin{cases}&amp;space;x_1&#39;&amp;space;=&amp;space;x_1&amp;space;x_2&amp;space;\\&amp;space;x_2&#39;&amp;space;=&amp;space;-x_1&amp;space;x_2^3&amp;space;\end{cases}}" width="200"/>

An example of quadratization of this system will be a new variable

<img alt="y = x_1 x_2^2" height="60" src="https://latex.codecogs.com/png.latex?\dpi{200}&amp;amp;amp;space;\huge&amp;amp;amp;space;{\color{DarkOrange}y&amp;amp;amp;space;=&amp;amp;amp;space;x_1&amp;amp;amp;space;x_2^2}" width="150"/>

leading to the following ODE

<img alt="y&#39; = x_2 y - 2y^2" height="50" src="https://latex.codecogs.com/png.latex?\dpi{200}&amp;space;\huge&amp;space;{\color{DarkOrange}y&#39;&amp;space;=&amp;space;x_2&amp;space;y&amp;space;-&amp;space;2y^2}" width="250"/>

Thus, we attained the system with quadratic right-hand side

<img alt="\begin{cases} x_1&#39; = x_1 x_2 \\ x_2&#39; = -x_2 y \\ y&#39; = x_2 y - 2y^2 \end{cases}" height="202" src="https://latex.codecogs.com/png.latex?\dpi{200}&amp;space;\huge&amp;space;{\color{DarkOrange}\begin{cases}&amp;space;x_1&#39;&amp;space;=&amp;space;x_1&amp;space;x_2&amp;space;\\&amp;space;x_2&#39;&amp;space;=&amp;space;-x_2&amp;space;y&amp;space;\\&amp;space;y&#39;&amp;space;=&amp;space;x_2&amp;space;y&amp;space;-&amp;space;2y^2&amp;space;\end{cases}}" width="300"/>

We used only one new variable, so we achieved an *optimal* quadratization.

# Qbee usage

QBee implements algorithms that **take** system of ODEs with elementary functions right-hand side and
**return** *optimal monomial quadratization* - optimal quadratization constructed from monomial substitutions.

We will demonstrate usage of QBee on the example below. Other interactive examples you can find
in [examples section](old_examples).

### 1. Importing QBee

```python
import sympy
from qbee import *

sympy.init_printing()  # If you work in Jupyter notebook 
```

### 2. Polynomial ODEs and quadratization

Our main focus is transforming systems of ODEs with polynomial right-hand side into quadratic ones. So, we will start
from discussing how to quadratize them.

First, we define a system of ODEs. In the list `system` defined right-hand side of input ODE in linear order. For
example, consider the system:

![\begin{cases} \dot{x} = y^3 \\ \dot{y} = x^3 \end{cases}](https://latex.codecogs.com/gif.latex?%5Chuge%20%5Cbegin%7Bcases%7D%20x%27%20%3D%20y%5E3%20%5C%5C%20y%27%20%3D%20x%5E3%20%5Cend%7Bcases%7D)

```python
x, y, _ = sympy.ring(['x', 'y'], sympy.QQ)
system = [
    y ** 3,
    x ** 3
]
```

Then we need to run quadratization algorithm. For the most systems high-level function `quadratize` should be enough.

```python
quad_system = quadratize(system)
print("Quadratized system:", quad_system)
```

Sample output:

```
Elapsed time: 0.011s.
Number of introduced variables: 3
Nodes traversed: 15
Introduced variables: ['x y', 'x^{2}', 'y^{2}']
Quadratized system: [w_3*y, w_2*x, w_2**2 + w_3**2, 2*w_1*w_3, 2*w_1*w_2]
```

Introduced variables are optimal monomial quadratization.

### 3. General ODEs and polynomialization

If given system of ODEs is not polynomial you can use our tools transform it into a one with polynomial right-hand side.

This functionality is writen in slightly different notation by now.

---

First, define variables and their derivatives
```python
x = sympy.symbols('x')
dx = derivatives(x)
```

---

Then we build a ODEs system with elementary function right-hand side

```python
system = EquationSystem([
    sympy.Eq(dx, 1 / (1 + sympy.exp(x)))
])
```

This system is not polynomial, so we need to polynomialize it.

---

We can convert equations right-hand side to polynomials

```python
poly_system = polynomialize(system)
poly_system.print()
```

Output:

```
x' = y_{1}
y_{0}' = y_{0}*y_{1}
y_{1}' = -y_{0}*y_{1}**3
```

**Note:** our implementation of polynomialization is **not** optimal yet.

### 4. Work inside of package

#### 1. Configuration

Inside of `config.ini` you can change the following arguments:

* `logging_enable = [True | False]`. If enabled, work of algorithm is logged into `logging_file` and `quad_systems_file`
  . Requires memory to work. Is not recommended for long quadratizations.
* `logging_file`: must be in Apache Arrow `.feather` format.
* `quad_systems_file`: dump quadratic systems by using pickle. `.pkl` file format is recommended.
* `progress_bar_enable`: enables progress bar during quadratization.

#### 2. Visualization

In order to visualize work of an algorithm you can pass logging data to `qbee.visualize.visualize_pyvis`:

```python
visualize_pyvis('log.feather', 'quad_systems.pkl')
```

## Papers

* Optimal Monomial Quadratization for ODE systems: [arxiv](https://arxiv.org/abs/2103.08013), [Springer](https://link.springer.com/chapter/10.1007/978-3-030-79987-8_9)

## Citation

If you find this code useful in your research, please consider citing the above paper that works best for you. 





