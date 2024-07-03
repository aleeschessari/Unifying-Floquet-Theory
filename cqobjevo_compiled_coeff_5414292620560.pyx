#!python
#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
cimport numpy as np
import scipy.special as spe
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.inter cimport _spline_complex_t_second, _spline_complex_cte_second
from qutip.cy.inter cimport _spline_float_t_second, _spline_float_cte_second
from qutip.cy.inter cimport _step_float_cte, _step_complex_cte
from qutip.cy.inter cimport _step_float_t, _step_complex_t
from qutip.cy.interpolate cimport (interp, zinterp)
from qutip.cy.cqobjevo_factor cimport StrCoeff
from qutip.cy.cqobjevo cimport CQobjEvo
from qutip.cy.math cimport erf, zerf
from qutip.qobj import Qobj
cdef double pi = 3.14159265358979323

include '/nobackup/my_python/lib/python3.8/site-packages/qutip/cy/complex_math.pxi'

cdef class CompiledStrCoeff(StrCoeff):
    cdef double w_d
    cdef double A_d

    def set_args(self, args):
        self.w_d=args['w_d']
        self.A_d=args['A_d']

    cdef void _call_core(self, double t, complex * coeff):
        cdef double w_d = self.w_d
        cdef double A_d = self.A_d

        coeff[0] = exp(-1j*w_d*t)
        coeff[1] = exp(1j*w_d*t)
        coeff[2] = exp(-1j*w_d*t)
        coeff[3] = exp(1j*w_d*t)
        coeff[4] = A_d
        coeff[5] = A_d*exp(-1j*2*w_d*t)
        coeff[6] = A_d*exp(1j*2*w_d*t)
        coeff[7] = w_d