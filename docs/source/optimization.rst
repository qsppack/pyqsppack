Optimization Module
===================

The optimization module provides functionality for Quantum Signal Processing (QSP) optimization.

QSP Solver
----------

.. currentmodule:: optimization.QSP_solver

.. autoclass:: optimization.QSP_solver
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: solve

Core Functionality
------------------

.. currentmodule:: optimization.core

.. autoclass:: optimization.core
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: get_unitary
.. autofunction:: get_unitary_sym
.. autofunction:: get_entry
.. autofunction:: reduced_to_full

Objective and Gradient Functions
--------------------------------

.. currentmodule:: optimization.objective

.. autoclass:: optimization.objective
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: obj_sym
.. autofunction:: grad_sym
.. autofunction:: grad_sym_real

Optimization Methods
--------------------

.. currentmodule:: optimization.optimizers

.. autoclass:: optimization.optimizers
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: lbfgs
.. autofunction:: coordinate_minimization
.. autofunction:: newton

Utility Functions
-----------------

.. currentmodule:: optimization.utils

.. autoclass:: optimization.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: chebyshev_to_func
.. autofunction:: cvx_poly_coef 