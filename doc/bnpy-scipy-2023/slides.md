---
theme: default
class: text-center
highlighter: shiki
lineNumbers: false
transition: slide-left
aspectRatio: 16/9
favicon: /favicon.ico
title: 'Out-Performing NumPy is Hard: When and How to Try with Your Own C-Extensions '
---

# Out-Performing NumPy is Hard: When and How to Try with Your Own C-Extensions

### Christopher Ariza

### CTO, Research Affiliates

<style>
h1 {font-size: 1.5em;}
</style>


---

# About Me

<Transform :scale="1.5">
<v-clicks>

CTO at Research Affiliates

Python programmer since 2000

PhD in music composition, professor of music technology

Python for algorithmic composition, computational musicology

Since 2012, builder of financial systems in Python

Creator of StaticFrame, an alternative DataFrame library
</v-clicks>
</Transform>


---
layout: center
---
# A passion for performance


---
---
# Python Performance

<Transform :scale="1.5">
<v-clicks>

Python (using C `PyObject`s) is relatively slow

C-extensions using C-types are fast

With NumPy, we get C-typed arrays in Python
</v-clicks>
</Transform>


---
---
# Can NumPy Routines be Optimized?

<Transform :scale="1.5">
<v-clicks>

Even with NumPy, performance opportunities remain

Some NumPy routines are implemented in Python

Many NumPy routines do more than we need
</v-clicks>
</Transform>


---
---
# Can NumPy Routines be Optimized?

<Transform :scale="1.5">
<v-clicks depth="2">

- NumPy routines are flexible
    - Handle N-dimensional arrays
    - Handle full diversity of dtypes
    - Handle non-array (i.e., list, tuple) inputs
- More narrow routines might be able to out-perform flexible routines

</v-clicks>
</Transform>


---
---
# Finding the First `True`

<Transform :scale="1.5">
<v-clicks>

A utility that was needed for StaticFrame

1D Boolean array: find the index of the first `True`

2D Boolean array: find the indices of the first `True` per axis

Search in both directions

Identify all `False`
</v-clicks>
</Transform>


---
layout: none
---
<div class="absolute top-0px">
<img src="/screen-so-1.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666;}
</style>


---
layout: none
---
<div class="absolute top-0px">
<img src="/screen-so-2.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666;}
</style>


---
layout: none
---
<div class="absolute top-0px">
<img src="/screen-gh-npy-issue-2269.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666;}
</style>
<!-- NumPy Issue 2269 -->


---
---
# Finding the First `True` with NumPy

<Transform :scale="1.5">
<v-clicks depth="2">

- No NumPy function does just what we need
- Two options are close
    - `np.argmax()`
    - `np.nonzero()`
</v-clicks>
</Transform>


---
---
# `np.argmax()`

<Transform :scale="1.5">
<v-clicks depth="3">

Return the index of the maximum value in an array

If there are ties, the first index is returned

Specialized for Boolean arrays to short-circuit on first `True`

Returns `0` if all `False`

Must call `np.any()` to discover all `False`
</v-clicks>
</Transform>


---
---
# `np.argmax()`
<Transform :scale="1.5">

```python {all|1|1-3|4-5|6-7}
>>> array = np.arange(10_000) == 2_000
>>> np.argmax(array) # finds first True
2000
>>> np.argmax(np.full(10_000, False)) # if all False, reports 0
0
>>> np.argmax(np.array([True, False])) # if True at index 0
0
```
</Transform>


---
---
# `np.nonzero()`

<Transform :scale="1.5">
<v-clicks depth="3">

Finds all non-zero positions

Returns a tuple of arrays per dimension

Cannot short-circuit
</v-clicks>
</Transform>


---
---
# `np.nonzero()`
<Transform :scale="1.5">

```python {all|1|1-3|1-5|6|6-8|6-10} {lines:false}
>>> array = np.arange(10_000) == 2_000
>>> np.nonzero(array)
(array([2000]),)
>>> np.nonzero(array)[0][0]
2000
>>> array = np.arange(10_000) % 2_000 == 0
>>> array.sum()
5
>>> np.nonzero(array)
(array([   0, 2000, 4000, 6000, 8000]),)
```
</Transform>


---
layout: center
---
# Performance of `np.argmax()` (with `np.any()`) and `np.nonzero()`


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-0.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
---
# Performance Panels

<Transform :scale="1.2">
<v-clicks depth="2">

- Bars in a plot are implementations (numbered and labelled in the legend)
- Rows are Boolean array size (1e5, 1e6, 1e7)
- Four columns show different fill characteristics
    - One `True`
        - Set at 1/3<sup>rd</sup> to the end
        - Set at 2/3<sup>rd</sup> to the end
    - 33% of size is `True`
        - Filled from 1/3<sup>rd</sup> to the end
        - Filled from 2/3<sup>rd</sup> to the end

</v-clicks>
</Transform>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-0.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
---
# Opportunities for Improvement

<Transform :scale="1.2">
<v-clicks depth="2">

- `np.argmax`:
    - All-`False` can be handled with `np.any()`
    - Worst case Requires two iterations, but can short-circuit
    - Does not search in reverse
- `np.nonzero`
    - Always requires one iteration, cannot short-circuit
    - Collects more than we need
    - Must iterate over results to find first or last
- Both options seem suboptimal

</v-clicks>
</Transform>

<!-- What if we write a C-extension to do just what we need
Only handle 1D, 2D contiguous Boolean arrays
Return -1 when all `False`
-->


---
---
# Many Options for Performance

<Transform :scale="1.5">
<v-clicks>

C-Extensions

Cython / Numba

Rust via PyO3
</v-clicks>
</Transform>
<!--
I will favor writing C-Extensions using the CPython C-API and NumPy C-API
-->


---
---
# Good Candidates for C-Implementation

<Transform :scale="1.5">
<v-clicks>

Core routine can be done without `PyObject`s

Can operate directly on a C array

Finding the first `True` in a C array is a good candidate
</v-clicks>
</Transform>


---
---
# ``first_true_1d()`` as a C Extension

<Transform :scale="1.5">
<v-clicks depth="2">

- A Function with Two Arguments
    - NumPy array
    - A Boolean (`True` for forward, `False` for reverse)
- Evaluate elements, return the index of the first `True`
- If no `True`, return `-1`
- Code: https://github.com/flexatone/np-bench

</v-clicks>
</Transform>


---
layout: center
---
# Defining a C extension


---
---
# A Minimal C Extension Module `np_bench`

```c {all|1-5|6-8,17|9|10-16}
static struct PyModuleDef npb_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "np_bench",
    .m_size = -1,
};
PyObject*
PyInit_np_bench(void)
{
    import_array();
    PyObject *m = PyModule_Create(&npb_module);
    if (!m || PyModule_AddStringConstant(m, "__version__", "0.1.0")
    ) {
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}
```


---
---
# A C Function as a Module-Level Python Function

```c {all|1-3,14|4-5|6-11|12-13}
static PyObject*
first_true_1d(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArrayObject *array = NULL;
    int forward = 1;
    if (!PyArg_ParseTuple(args,
            "O!p:first_true_1d",
            &PyArray_Type, &array,
            &forward)) {
        return NULL;
    }
    // implmentation
    return PyLong_FromSsize_t(-1);
}
```

---
---
# Adding a C Function to a Python Module

```c {all|1-4|5-10}
static PyMethodDef npb_methods[] =  {
    {"first_true_1d", (PyCFunction)first_true_1d, METH_VARARGS, NULL},
    {NULL},
};
static struct PyModuleDef npb_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "np_bench",
    .m_size = -1,
    .m_methods = npb_methods,
};
```

---
layout: center
---
# Reading elements from an array in C


---
---
# Reading Elements from an Array in C

<Transform :scale="1.5">
<v-clicks>

1. Reading Native `PyObject`s From Arrays (``PyArray_GETITEM``)
1. Reading NumPy Scalar `PyObject`s From Arrays (``PyArray_ToScalar``)
1. Casting Data Pointers to C-Types (``PyArray_GETPTR1``)
1. Using `NpyIter`
1. Using C-Arrays and Pointer Arithmetic (``PyArray_DATA()``)
</v-clicks>
</Transform>


---
layout: center
---
# Working with `PyObject`s


---
---
# I: Reading Native `PyObject`s From Arrays

<Transform :scale="1.5">
<v-clicks>

Only process 1D arrays

Use `PyArray_GETPTR1()` to get pointer to element

Use `PyArray_GETITEM()` to build corresponding `PyObject`

Use Python C-API `PyObject_IsTrue()` to evaluate element

Must manage reference counts for `PyObject`s
</v-clicks>
</Transform>


---
---
# I: Reading Native `PyObject`s From Arrays

<Transform :scale="1.1">
```c {all|1-3,17|4-5|6-11|12-16}
static PyObject*
first_true_1d_getitem(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArrayObject *array = NULL;
    int forward = 1;
    if (!PyArg_ParseTuple(args,
            "O!p:first_true_1d_getitem",
            &PyArray_Type, &array,
            &forward)) {
        return NULL;
    }
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    // ... implementation
}
```
</Transform>


---
---
# I: Reading Native `PyObject`s From Arrays

<Transform :scale="1.1">
```c {all|1-3|4,13|5,12|6|7-11}
    npy_intp size = PyArray_SIZE(array);
    npy_intp i;
    PyObject* element;
    if (forward) {
        for (i = 0; i < size; i++) {
            element = PyArray_GETITEM(array, PyArray_GETPTR1(array, i));
            if(PyObject_IsTrue(element)) {
                Py_DECREF(element);
                break;
            }
            Py_DECREF(element);
        }
    }
```
</Transform>

---
---
# I: Reading Native `PyObject`s From Arrays

<Transform :scale="1.1">

```c {all|1,10|2,9|3|4-8|11-14}
    else { // reverse
        for (i = size - 1; i >= 0; i--) {
            element = PyArray_GETITEM(array, PyArray_GETPTR1(array, i));
            if(PyObject_IsTrue(element)) {
                Py_DECREF(element);
                break;
            }
            Py_DECREF(element);
        }
    }
    if (i < 0 || i >= size ) {
        i = -1;
    }
    return PyLong_FromSsize_t(i);
```
</Transform>

---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-1.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
layout: center
---
# What about NumPy scalars?


---
---
# II: Reading NumPy Scalar `PyObject`s From Arrays

<Transform :scale="1.5">
<v-clicks>

Only process 1D arrays

Use `PyArray_GETPTR1()` to get pointer to element

Use `PyArray_ToScalar()` to build NumPy scalar `PyObject`

Can continue to use `PyObject_IsTrue()` to evaluate elements

Must manage reference counting for `PyObject`s
</v-clicks>
</Transform>



---
---
# II: Reading NumPy Scalar `PyObject`s From Arrays

<Transform :scale="1.1">

```c {all|1-3,10|4|5-8|9}
static PyObject*
first_true_1d_scalar(PyObject *Py_UNUSED(m), PyObject *args)
{
    // ... parse args
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    // ... implementation
}
```
</Transform>


---
---
# II: Reading NumPy Scalar `PyObject`s From Arrays
<Transform :scale="1.2">

```c {all|1-3|5,14|6,13|7|8-12}
    npy_intp size = PyArray_SIZE(array);
    npy_intp i;
    PyObject* scalar;

    if (forward) {
        for (i = 0; i < size; i++) {
            scalar = PyArray_ToScalar(PyArray_GETPTR1(array, i), array);
            if(PyObject_IsTrue(scalar)) {
                Py_DECREF(scalar);
                break;
            }
            Py_DECREF(scalar);
        }
    }
```
</Transform>

---
---
# II: Reading NumPy Scalar `PyObject`s From Arrays
<Transform :scale="1.1">

```c {all|1,10|2,9|3|4-8|11-14}
    else {
        for (i = size - 1; i >= 0; i--) {
            scalar = PyArray_ToScalar(PyArray_GETPTR1(array, i), array);
            if(PyObject_IsTrue(scalar)) {
                Py_DECREF(scalar);
                break;
            }
            Py_DECREF(scalar);
        }
    }
    if (i < 0 || i >= size ) {
        i = -1;
    }
    return PyLong_FromSsize_t(i);
```
</Transform>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-2.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
layout: center
---
# Avoiding `PyObject`s entirely


---
---
# III: Casting Data Pointers to C-Types

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean arrays

Use `PyArray_GETPTR1()` and cast to C type

No use of Python C-API, no reference counting

Can release the GIL over core loop
</v-clicks>
</Transform>


---
---
# III: Casting Data Pointers to C-Types
<Transform :scale="1.1">

```c {all|1-3,14|4|5-8|9-12|13}
static PyObject*
first_true_1d_getptr(PyObject *Py_UNUSED(m), PyObject *args)
{
    // ... parse args
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    // ... implementation
}
```
</Transform>


---
---
# III: Casting Data Pointers to C-Types
<Transform :scale="1.1">

```c {all|1-2|4,10|5,9|6-8}
    npy_intp size = PyArray_SIZE(array);
    npy_intp i;

    if (forward) {
        for (i = 0; i < size; i++) {
            if(*(npy_bool*)PyArray_GETPTR1(array, i)) {
                break;
            }
        }
    }
```
</Transform>


---
---
# III: Casting Data Pointers to C-Types
<Transform :scale="1.1">

```c {all|1,7|2,6|3-5|8-11}
    else { // reverse
        for (i = size - 1; i >= 0; i--) {
            if(*(npy_bool*)PyArray_GETPTR1(array, i)) {
                break;
            }
        }
    }
    if (i < 0 || i >= size ) {
        i = -1;
    }
    return PyLong_FromSsize_t(i);
```
</Transform>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-3.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-4.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
layout: center
---
# Other options with the NumPy C API


---
---
# IV: Using `NpyIter`

<Transform :scale="1.5">
<v-clicks>

Requires `NpyIter_New()` and related library functions

Generality for N-dimensional arrays of diverse homogeneity

Perform stride-sized pointer arithmetic in inner loop

Requires more code

Does not support reverse iteration
</v-clicks>
</Transform>


---
---
# IV: Using `NpyIter`
<Transform :scale="1.1">

```c {all|1-3,14|4|5-8|9-12|13}
static PyObject*
first_true_1d_npyiter(PyObject *Py_UNUSED(m), PyObject *args)
{
    // ... parse args
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    // ... implementation
}
```
</Transform>


---
---
# IV: Using `NpyIter`
<Transform :scale="1.1">

```c {all|1-7|8-10|11-15}
    NpyIter *iter = NpyIter_New(
            array,                                      // array
            NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP, // iter flags
            NPY_KEEPORDER,                              // order
            NPY_NO_CASTING,                             // casting
            NULL                                        // dtype
            );
    if (iter == NULL) {
        return NULL;
    }
    NpyIter_IterNextFunc *iter_next = NpyIter_GetIterNext(iter, NULL);
    if (iter_next == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }
```
</Transform>

---
---
# IV: Using `NpyIter`
<Transform :scale="1.1">

```c {all|1-2|4-5|7-8|10}
    npy_bool **data_ptr_array = (npy_bool**)NpyIter_GetDataPtrArray(iter);
    npy_bool *data_ptr;

    npy_intp *stride_ptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp stride;

    npy_intp *inner_size_ptr = NpyIter_GetInnerLoopSizePtr(iter);
    npy_intp inner_size;

    npy_intp i = 0;
```
</Transform>

---
---
# IV: Using `NpyIter`
<Transform :scale="1.1">

```c {all|1,12|2-4|5,11|6-8,16-18|9-10|13-18}
    do {
        data_ptr = *data_ptr_array;
        stride = *stride_ptr;
        inner_size = *inner_size_ptr;
        while (inner_size--) {
            if (*data_ptr) {
                goto exit;
            }
            i++;
            data_ptr += stride;
        }
    } while(iter_next(iter));
    if (i == PyArray_SIZE(array)) {
        i = -1;
    }
exit:
    NpyIter_Deallocate(iter);
    return PyLong_FromSsize_t(i);
```
</Transform>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-5.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
layout: center
---
# Assuming array contiguity...


---
---
# V(a.): Using C-Arrays and Pointer Arithmetic

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean, and *contiguous* arrays

Use `PyArray_DATA()` to get pointer to underlying C-array

Advance through array with pointer arithmetic
</v-clicks>
</Transform>


---
---
# V(a.): Using C-Arrays and Pointer Arithmetic
<Transform :scale="1.1">

```c {all|1-3,18|4|5-8|9-12|13-16|17}
static PyObject*
first_true_1d_ptr(PyObject *Py_UNUSED(m), PyObject *args)
{
    // ... parse args
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(array)) {
        PyErr_SetString(PyExc_ValueError, "Array must be contiguous");
        return NULL;
    }
    // ... implementation
}
```
</Transform>

---
---
# V(a.): Using C-Arrays and Pointer Arithmetic
<Transform :scale="1.1">

```c {all|1|3-6|8,17|9,10|11,16|12-15}
    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    npy_intp size = PyArray_SIZE(array);
    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;

    if (forward) {
        p = array_buffer;
        p_end = p + size;
        while (p < p_end) {
            if (*p) {
                break;
            }
            p++;
        }
    }
```
</Transform>


---
---
# V(a.): Using C-Arrays and Pointer Arithmetic
<Transform :scale="1.1">

```c {all|1,10|2-3|4,9|5-8|11-14}
    else { // reverse
        p = array_buffer + size - 1;
        p_end = array_buffer - 1;
        while (p > p_end) {
            if (*p) {
                break;
            }
            p--;
        }
    }
    if (p != p_end) {
        position = p - array_buffer;
    }
    return PyLong_FromSsize_t(position);
```
</Transform>


---
layout: center
---
# This must be the fastest approach...


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-6.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>



---
layout: center
---
# How is ``np.argmax()`` still faster?


---
---
# Performance Beyond Contiguous Iteration

<Transform :scale="1.5">
<v-clicks depth="2">

Single instruction, multiple data (SIMD) instructions

... via CPU dispatch `NPY_CPU_DISPATCH_CALL_XB`
</v-clicks>
</Transform>


---
---
# NumPy SIMD `BOOL_argmax`
<Transform :scale="0.8">

```c {all|1-3,25|4-7|8,23|9-12|13-16|17-18|19-24}
NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(BOOL_argmax)
(npy_bool *ip, npy_intp len, npy_intp *mindx, PyArrayObject *NPY_UNUSED(aip))
{
    npy_intp i = 0;
    const npyv_u8 zero = npyv_zero_u8();
    const int vstep = npyv_nlanes_u8;
    const int wstep = vstep * 4;
    for (npy_intp n = len & -wstep; i < n; i += wstep) {
        npyv_u8 a = npyv_load_u8(ip + i + vstep*0);
        npyv_u8 b = npyv_load_u8(ip + i + vstep*1);
        npyv_u8 c = npyv_load_u8(ip + i + vstep*2);
        npyv_u8 d = npyv_load_u8(ip + i + vstep*3);
        npyv_b8 m_a = npyv_cmpeq_u8(a, zero);
        npyv_b8 m_b = npyv_cmpeq_u8(b, zero);
        npyv_b8 m_c = npyv_cmpeq_u8(c, zero);
        npyv_b8 m_d = npyv_cmpeq_u8(d, zero);
        npyv_b8 m_ab = npyv_and_b8(m_a, m_b);
        npyv_b8 m_cd = npyv_and_b8(m_c, m_d);
        npy_uint64 m = npyv_tobits_b8(npyv_and_b8(m_ab, m_cd));
        if ((npy_int64)m != ((1LL << vstep) - 1)) { // if not all zero
            break;
        }
    }
    // ... element-wise evaluate from current i to the end
}
```
</Transform>
<!--
From: numpy/core/src/multiarray/argfunc.dispatch.c.src
See also: misc.h
numpy/core/src/_simd/_simd_inc.h.src
// convert boolean vector to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
 -->

---
---
# Performance Beyond Contiguous Iteration

<Transform :scale="1.5">
<v-clicks depth="2">

- SIMD is hard in C
- SIMD reduces loop iteration
- Use loop unrolling
    - Reduce `for`-loop iterations
    - Increase branch prediction

</v-clicks>
</Transform>


---
---
# V(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean, contiguous arrays

Use `PyArray_DATA()` to get C-array

Advance through array with pointer arithmetic, unrolling units of 4
</v-clicks>
</Transform>


---
---
# V(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
<Transform :scale="1.0">

```c {all|1-3,18|4|5-8|9-12|13-16|17}
static PyObject*
first_true_1d_ptr_unroll(PyObject *Py_UNUSED(m), PyObject *args)
{
    // ... parse args
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(array)) {
        PyErr_SetString(PyExc_ValueError, "Array must be contiguous");
        return NULL;
    }
    // ... implementation
}
```
</Transform>


---
---
# V(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
<Transform :scale="1.1">

```c {all|1|3-4|6-9}
    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    npy_intp size = PyArray_SIZE(array);
    lldiv_t size_div = lldiv((long long)size, 4); // unroll 4 iterations

    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;
```
</Transform>



---
---
# V(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
<Transform :scale="1.0">

```c {all|1,18|2-3|4,13|5-12|14,17|15-16}
    if (forward) {
        p = array_buffer;
        p_end = p + size;
        while (p < p_end - size_div.rem) {
            if (*p) {break;}
            p++;
            if (*p) {break;}
            p++;
            if (*p) {break;}
            p++;
            if (*p) {break;}
            p++;
        }
        while (p < p_end) {
            if (*p) {break;}
            p++;
        }
    }
```
</Transform>


---
---
# V(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
<Transform :scale="1.0">

```c {all|1,18|2-3|4,13|5-12|14,17|15-16}
    else { // reverse
        p = array_buffer + size - 1;
        p_end = array_buffer - 1;
        while (p > p_end + size_div.rem) {
            if (*p) {break;}
            p--;
            if (*p) {break;}
            p--;
            if (*p) {break;}
            p--;
            if (*p) {break;}
            p--;
        }
        while (p > p_end) {
            if (*p) {break;}
            p--;
        }
    }
```
</Transform>


---
---
# V(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
<Transform :scale="1.1">

```c {all}
    if (p != p_end) {
        position = p - array_buffer;
    }
    return PyLong_FromSsize_t(position);
```
</Transform>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-7.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
---
# Performance Beyond Contiguous Iteration

<Transform :scale="1.5">
<v-clicks depth="2">

- SIMD used to look ahead for `True`
- Use `memcmp()`
    - Compare raw memory to zero array buffer
    - `memcmp()` might use SIMD

</v-clicks>
</Transform>


---
---
# V(c.): Using C-Arrays, `memcmp()` Scan

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean, contiguous arrays

Use `PyArray_DATA()` to get C-array

Lookhead in units of 8 bytes

Less code than loop unrolling
</v-clicks>
</Transform>


---
---
# V(c.): Using C-Arrays, `memcmp()` Scan
<Transform :scale="1.1">

```c {all|1|2-4,19|5|6-9|10-13|14-17|18}
#define MEMCMP_SIZE 8
static PyObject*
first_true_1d_memcmp(PyObject *Py_UNUSED(m), PyObject *args)
{
    // ... parse args
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(array)) {
        PyErr_SetString(PyExc_ValueError, "Array must be contiguous");
        return NULL;
    }
    // ... implementation
}
```
</Transform>



---
---
# V(c.): Using C-Arrays, `memcmp()` Scan
<Transform :scale="1.1">

```c {all|1|2|4-5|7-9}
    static npy_bool zero_buffer[MEMCMP_SIZE] = {0};
    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    npy_intp size = PyArray_SIZE(array);
    lldiv_t size_div = lldiv((long long)size, MEMCMP_SIZE); // quot, rem

    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;
```
</Transform>


---
---
# V(c.): Using C-Arrays, `memcmp()` Scan
<Transform :scale="1.1">

```c {all|1,14|2-3|4,9|5-8|10-13}
    if (forward) {
        p = array_buffer;
        p_end = p + size;
        while (p < p_end - size_div.rem) {
            if (memcmp(p, zero_buffer, MEMCMP_SIZE) != 0) {
                break;
            }
            p += MEMCMP_SIZE;
        }
        while (p < p_end) {
            if (*p) {break;}
            p++;
        }
    }
```
</Transform>

---
---
# V(c.): Using C-Arrays, `memcmp()` Scan
<Transform :scale="1.1">

```c {all|1,14|2-3|4,9|5-8|10-13|15-18}
    else { // reverse
        p = array_buffer + size - 1;
        p_end = array_buffer - 1;
        while (p > p_end + size_div.rem) {
            if (memcmp(p - MEMCMP_SIZE + 1, zero_buffer, MEMCMP_SIZE) != 0) {
                break;
            }
            p -= MEMCMP_SIZE;
        }
        while (p > p_end) {
            if (*p) {break;}
            p--;
        }
    }
    if (p != p_end) {
        position = p - array_buffer;
    }
    return PyLong_FromSsize_t(position);
```
</Transform>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-8.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
layout: center
---
# The big picture



---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-9.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
layout: center
---
# What about 2D?


---
---
# `first_true_2d()`

<Transform :scale="1.2">
<v-clicks depth="2">

- Deliver results by axis
    - Axis 0 returns an array of columns length
    - Axis 1 returns an array of rows length
- Might be C or Fortran contiguous
- If C-contiguous and Axis 1
    - Use pointer arithmetic (and `memcmp` scanning) through each row
    - Short-circuit if `True` found, jump to next row
- F-contiguous and Axis 0 works the same
- Use `PyArray_NewCopy()` to get new contiguous ordering

</v-clicks>
</Transform>


---
layout: center
---
# Can we outfperform `np.argmax()` in 2D?


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft2d-fig-0.png" style="height: 550px;" />
</div>

<style>
div {background-color: #fff;}
</style>


---
---
# Reflections

<Transform :scale="1.5">
<v-clicks depth="2">

- Recognize when the work can be done with C-types
- Implement limited functions
    - Only support dimensionality needed
    - Specialized 1D and 2D are most practical
    - Only support needed dtypes
    - Require contiguity when appropriate
- Constantly test performance

</v-clicks>
</Transform>


---
---
# Thanks!

<Transform :scale="1.5">

Sli.dev slide toolkit

Code: https://github.com/flexatone/np-bench

`first_true_1d`, `first_true_2d` packaged: https://pypi.org/project/arraykit

StaticFrame: https://static-frame.dev

</Transform>
