# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:34:54 2022

@author: Admin
"""
from scipy.ndimage import _ni_support
import numpy as np
from numpy.core.multiarray import normalize_axis_index
from scipy.ndimage import _nd_image

def _invalid_origin(origin, lenw):
    return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)

def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input and complex_weights:
        # real component of the output
        func(input.real, weights.real, output=output.real,
             cval=np.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None,
                            cval=np.imag(cval), **kwargs)
        # imaginary component of the output
        func(input.real, weights.imag, output=output.imag,
             cval=np.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None,
                            cval=np.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=np.real(cval),
             **kwargs)
        func(input.imag, weights, output=output.imag, cval=np.imag(cval),
             **kwargs)
    else:
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the "
                             "input is real.")
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    return output

def correlate1d(input, weights, axis=-1, output=None, mode="reflect",
                cval=0.0, origin=0):
    """Calculate a 1-D correlation along the given axis.

    The lines of the array along the given axis are correlated with the
    given weights.

    Parameters
    ----------
    %(input)s
    weights : array
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
    input = np.asarray(input)
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input or complex_weights:
        if complex_weights:
            # print("==========Complex Weight============")
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        kwargs = dict(axis=axis, mode=mode, origin=origin)
        output = _ni_support._get_output(output, input, complex_output=True)
        return _complex_via_real_components(correlate1d, input, weights,
                                            output, cval, **kwargs)

    output = _ni_support._get_output(output, input)
    # output = np.zeros(input.shape)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = normalize_axis_index(axis, input.ndim)
    # Axis error if (-ndim <= axis < ndim) is false
    # ex: normalize_axis_index(0, 3) -> return 0
    # ex: normalize_axis_index(1, 3) -> return 1
    if _invalid_origin(origin, len(weights)):
        raise ValueError('Invalid origin; origin must satisfy '
                         '-(len(weights) // 2) <= origin <= '
                         '(len(weights)-1) // 2')
    mode = _ni_support._extend_mode_to_code(mode)
    # mode 'constant' -> 4
    _nd_image.correlate1d(input, weights, axis, output, mode, cval,
                          origin)
    return output

def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    # exponent_range = [0]
    sigma2 = sigma * sigma
    # # print("========RADIUS==========")
    # print(radius)
    x = np.arange(-radius, radius+1)
    # print("========RADIUS RANGE===========")
    # print(x)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    # phi_x = [exp(x[0]), exp(x[1]), exp(x[2])]
    # np.exp(x) -> e^x -> e = 2.718281 -> value is always positive -> ex : 2^-2 = 1/(2^2) = 1/4
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x
    
    
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    """1-D Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode_reflect)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    gaussian_filter1d : ndarray

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter1d
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(101).cumsum()
    >>> y3 = gaussian_filter1d(x, 3)
    >>> y6 = gaussian_filter1d(x, 6)
    >>> plt.plot(x, 'k', label='original data')
    >>> plt.plot(y3, '--', label='filtered, sigma=3')
    >>> plt.plot(y6, ':', label='filtered, sigma=6')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.show()

    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # lw = int(4.0 * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    # print("=======WEIGHT=========")
    # print(weights)
    # [::-1] means revert kernel -> ex: np.array([1,2,3])[:-1] -> array([3,2,1])
    return correlate1d(input, weights, axis, output, mode, cval, 0)

def gaussian_filter(input, sigma, nonZeroIndex=None, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    """Multidimensional Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : int or sequence of ints, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. A positive order
        corresponds to convolution with that derivative of a Gaussian.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    truncate : float
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    gaussian_filter : ndarray
        Returned array of same shape as `input`.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter
    >>> a = np.arange(50, step=2).reshape((5,5))
    >>> a
    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28],
           [30, 32, 34, 36, 38],
           [40, 42, 44, 46, 48]])
    >>> gaussian_filter(a, sigma=1)
    array([[ 4,  6,  8,  9, 11],
           [10, 12, 14, 15, 17],
           [20, 22, 24, 25, 27],
           [29, 31, 33, 34, 36],
           [35, 37, 39, 40, 42]])

    >>> from scipy import misc
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = misc.ascent()
    >>> result = gaussian_filter(ascent, sigma=5)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)
    if (nonZeroIndex == None):
        nonZeroIndex = list(zip(np.nonzero(input)[0], np.nonzero(input)[1]))
    output = _ni_support._get_output(output, input)
    # output will be np.zeros(input.shape)
    # print("=======OUPUT======")
    # print(output)
    orders = _ni_support._normalize_sequence(order, input.ndim)
    # _normalize_sequence(0, 2) --> [0,0]
    # input.ndim --> dimension of input, ex : np.array([[1,2,3]]) -> 2
    # order determines the formula of gaussian used
    # 0 -> using base gaussian formula
    # 1 -> using derivatives 1 of gaussian
    # print("=======ORDERS======")
    # print(orders)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    # sigmas --> [sigma, sigma]
    # print("=======SIGMA======")
    # print(sigmas)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    # modes --> ['constant', 'constant']
    # print("=======MODES======")
    # print(modes)
    axes = list(range(input.ndim))
    # axes = [0,1]
    # print("=======AXIS======")
    # print(axes)
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    # axes = [(0, sigma, order, mode), (1, sigma, order, mode)]
    
    result = None
    nonZeroIndexOfResult = None
    if len(axes) == 2:
        i = 0
        for axis, sigma, order, mode in axes:
            if (i == 0):
                newInput = np.take(input, [nonZeroIndex[0][0]], axis=1)
                result = gaussian_filter1d(newInput, sigma, axis, order, None,
                                  mode, cval, truncate)
                nonZeroIndexOfResult = list(zip(np.nonzero(result)[0], 
                                                np.nonzero(result)[1]))
            else :
                newInput = np.zeros((len(nonZeroIndexOfResult), 
                                     input.shape[1]))
                k = 0
                for (row, col) in nonZeroIndexOfResult:
                    newInput[k][nonZeroIndex[0][1]] = result[row][col]
                    k = k + 1
                result = gaussian_filter1d(newInput, sigma, axis, order, None,
                                  mode, cval, truncate)
            # print("=======OUTPUT RES======")
            # print(output)
            input = output
            i = i + 1
    else:
        output[...] = input[...]
    
    i = 0
    for (row, col) in nonZeroIndexOfResult:
        output[row] = result[i]
        i = i + 1
    return output
