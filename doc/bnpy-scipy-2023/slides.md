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
# A passion for Python performance


---
---
# Python Performance

<Transform :scale="1.5">
<v-clicks depth="2">

- Python is relatively slow
    - All values are "boxed" in C `PyObject`s
    - Values not in contiguous memory
    - Must manage reference counts
- C-extensions using C-types are fast
- With NumPy, we get C-typed arrays in Python
</v-clicks>
</Transform>


---
---
# Can NumPy Routines be Optimized?

<Transform :scale="1.5">
<v-clicks>

1. Some NumPy routines are implemented in Python
2. Many NumPy routines do more than we need

</v-clicks>
</Transform>

---
---
# Optimizing NumPy Routines written in Python

<Transform :scale="1.5">
<v-clicks depth="2">

- Some NumPy routines are implemented in Python
    - `np.roll()`
    - `np.linspace()`
- All leverage lower-level C routines
- Little chance a C-implementation will be faster

</v-clicks>
</Transform>



---
---
# Optimizing Excessively Flexible NumPy Routines

<Transform :scale="1.5">
<v-clicks depth="2">

- NumPy routines are flexible
    - Handle N-dimensional arrays
    - Handle full diversity of dtypes
    - Support diverse array memory layouts (non-contiguous memory)
    - Handle non-array (i.e., list, tuple) inputs
- Flexibility has a performance cost
- More narrow routines might be more efficient

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

All `False` returns an ambigous `0`

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
# Performance of `np.argmax()` + `np.any()` & `np.nonzero()`


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-0.png" style="height: 550px;" />
</div>

<style>
div {background-color: #d5d0ce;}
</style>


---
---
# Performance Panels

<Transform :scale="1.5">
<v-clicks depth="2">

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
div {background-color: #d5d0ce;}
</style>


---
---
# Opportunities for Improvement

<Transform :scale="1.5">
<v-clicks depth="2">

- `np.argmax()`
    - Must call `np.any()` to discover all-`False`
    - Worst case requires two iterations, but can short-circuit
    - Does not search in reverse
- `np.nonzero()`
    - Requires one full iteration (cannot short-circuit)
    - Collects more than we need
    - Must iterate over results to find first or last
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

C extensions

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
</v-clicks>
</Transform>


---
---
# ``first_true_1d()`` as a C Extension

<Transform :scale="1.5">
<v-clicks depth="2">

- A function with two arguments
    - NumPy array
    - A Boolean (`True` for forward, `False` for reverse)
- Evaluate each element, return the index of the first `True`
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
<Transform :scale="1.1">

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
</Transform>


---
---
# A C Function as a Module-Level Python Function
<Transform :scale="1.1">

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
</Transform>


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
# Four Ways to Read Elements

<Transform :scale="1.5">
<v-clicks>

1. Reading Native `PyObject`s From Arrays (``PyArray_GETITEM``)
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

```c {all|1-3,10|4|5-8|9}
static PyObject*
first_true_1d_getitem(PyObject *Py_UNUSED(m), PyObject *args)
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
div {background-color: #d5d0ce;}
</style>



---
layout: center
---
# Use C types instead of `PyObject`s


---
---
# II: Casting Data Pointers to C-Types

<Transform :scale="1.5">
<v-clicks>

Only process 1D, *Boolean* arrays

Use `PyArray_GETPTR1()` and cast to C type

No use of Python C-API, no reference counting

Can release the GIL over core loop
</v-clicks>
</Transform>


---
---
# II: Casting Data Pointers to C-Types
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
# II: Casting Data Pointers to C-Types
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
# II: Casting Data Pointers to C-Types
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
div {background-color: #d5d0ce;}
</style>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-4.png" style="height: 550px;" />
</div>

<style>
div {background-color: #d5d0ce;}
</style>


---
layout: center
---
# Other options within the NumPy C API


---
---
# III: Using `NpyIter`

<Transform :scale="1.5">
<v-clicks>

`NpyIter` provides common iteration interface in C

Supports all dimensionalities, dtypes, memory layouts

Performs stride-sized pointer arithmetic in inner loop

Requires more code

Does not support reverse iteration
</v-clicks>
</Transform>


---
---
# III: Using `NpyIter`
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
# III: Using `NpyIter`
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
# III: Using `NpyIter`
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
# III: Using `NpyIter`
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
div {background-color: #d5d0ce;}
</style>


---
layout: center
---
# Assuming array contiguity...


---
---
# IV(a.): Using C-Arrays and Pointer Arithmetic

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean, and *contiguous* arrays

Use `PyArray_DATA()` to get pointer to underlying C-array

Advance through array with pointer arithmetic
</v-clicks>
</Transform>


---
---
# IV(a.): Using C-Arrays and Pointer Arithmetic
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
# IV(a.): Using C-Arrays and Pointer Arithmetic
<Transform :scale="1.1">

```c {all|1|3-6|8,17|9,10|11,16|12-15}
    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    npy_intp size = PyArray_SIZE(array);
    Py_ssize_t i = -1;
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
# IV(a.): Using C-Arrays and Pointer Arithmetic
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
        i = p - array_buffer;
    }
    return PyLong_FromSsize_t(i);
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
div {background-color: #d5d0ce;}
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

SSE SIMD on x86-64 has 128 bit registers

16 1-byte Booleans can be processed in one instruction

AVX-512 permits processing 512 bit registers

True vectorization

CPU dispatching permits usage when available
</v-clicks>
</Transform>


---
---
# NumPy SIMD `BOOL_argmax`
<Transform :scale="0.8">

```c {all|1-3,25|4-7|8,23|9-19|20-22,24}
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
# NumPy SIMD `BOOL_argmax`
<Transform :scale="1.4">

```c {all|1-4|5-8|9-10|11-14}
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
```
</Transform>


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
# IV(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean, contiguous arrays

Use `PyArray_DATA()` to get C-array

Advance through array with pointer arithmetic, unrolling units of 4
</v-clicks>
</Transform>


---
---
# IV(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
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
# IV(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
<Transform :scale="1.1">

```c {all|1|3-4|6-9}
    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    npy_intp size = PyArray_SIZE(array);
    lldiv_t size_div = lldiIv((long long)size, 4); // unroll 4 iterations

    Py_ssize_t i = -1;
    npy_bool *p;
    npy_bool *p_end;
```
</Transform>



---
---
# IV(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
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
# IV(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
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
# IV(b.): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
<Transform :scale="1.1">

```c {all}
    if (p != p_end) {
        i = p - array_buffer;
    }
    return PyLong_FromSsize_t(i);
```
</Transform>


---
layout: none
---
<div class="absolute top-0px">
<img src="/ft1d-fig-7.png" style="height: 550px;" />
</div>

<style>
div {background-color: #d5d0ce;}
</style>


---
---
# Performance Beyond Contiguous Iteration

<Transform :scale="1.5">
<v-clicks depth="2">

- SIMD used to look ahead for `True`
- Use `memcmp()` compare raw memory to a zero array buffer
- Can cast 8 bytes of memory to `npy_uint64` and compare to `0`

</v-clicks>
</Transform>


---
---
# IV(c.): Using C-Arrays, Forward Scan

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean, contiguous arrays

Use `PyArray_DATA()` to get C-array

Forward scanning in units of 8 bytes

Less code than loop unrolling
</v-clicks>
</Transform>


---
---
# IV(c.): Using C-Arrays, Forward Scan
<Transform :scale="1.1">

```c {all|1-3,18|4|5-8|9-12|13-16|17}
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
# IV(c.): Using C-Arrays, Forward Scan
<Transform :scale="1.1">

```c {all|1|2|4-5|7-9}
    npy_intp lookahead = sizeof(npy_uint64);
    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    npy_intp size = PyArray_SIZE(array);
    lldiv_t size_div = lldiv((long long)size, lookahead); // quot, rem

    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;
```
</Transform>


---
---
# IV(c.): Using C-Arrays, Forward Scan
<Transform :scale="1.1">

```c {all|1,14|2-3|4,9|5-8|10-13}
    if (forward) {
        p = array_buffer;
        p_end = p + size;
        while (p < p_end - size_div.rem) {
            if (*(npy_uint64*)p != 0) {
                break;
            }
            p += lookahead;
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
# IV(c.): Using C-Arrays, Forward Scan
<Transform :scale="1.1">

```c {all|1,14|2-3|4,9|5-8|10-13|15-18}
    else { // reverse
        p = array_buffer + size - 1;
        p_end = array_buffer - 1;
        while (p > p_end + size_div.rem) {
            if (*(npy_uint64*)(p - lookahead + 1) != 0) {
                break;
            }
            p -= lookahead;
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
div {background-color: #d5d0ce;}
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
div {background-color: #d5d0ce;}
</style>


---
layout: center
---
# Out-performing NumPy is hard!



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
    - Require contiguity when possible
- Constantly test performance
</v-clicks>
</Transform>


---
---
# Thank You

<Transform :scale="1.5">

Sli.dev slide toolkit

Code: https://github.com/flexatone/np-bench

`first_true_1d`, `first_true_2d` packaged: https://pypi.org/project/arraykit

StaticFrame: https://static-frame.dev
</Transform>
