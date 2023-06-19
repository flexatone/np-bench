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
# Optimizing Python and NumPy Usage

<Transform :scale="1.5">
<v-clicks>

Even with NumPy, performance opportunities remain

Custom NumPy array builders (`delimited_to_arrays()`)

Replacing non-array `PyObject` usage with C-types (`BlockIndex`)

What about NumPy routines?
</v-clicks>
</Transform>


---
---
# Can NumPy Routines be Optimized?

<Transform :scale="1.5">
<v-clicks>

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
# Finding the First True with NumPy

<Transform :scale="2">
<v-clicks depth="3">

`np.argmax()`

`np.nonzero()`
</v-clicks>
</Transform>


---
---
# `np.argmax()`

<Transform :scale="1.5">
<v-clicks depth="3">

Return the index of the maximum value in an array

Specialized for Boolean arrays to short-circuit

Returns `0` if all `False`
</v-clicks>
</Transform>


---
---
# `np.argmax()` 1D
<Transform :scale="1.5">

```python {all|1|1-3|4|4-6}
>>> array = np.arange(10_000) == 2_000
>>> np.argmax(array) # finds first True
2000
>>> array = np.full(10_000, False)
>>> np.argmax(array) # if all False, reports 0
0
```
</Transform>


---
---
# `np.argmax()` 2D
<Transform :scale="1.5">

```python {all|1|1-6|7-8|9-10}
>>> array = np.arange(24).reshape(4,6) % 5 == 0
>>> array
array([[ True, False, False, False, False,  True],
       [False, False, False, False,  True, False],
       [False, False, False,  True, False, False],
       [False, False,  True, False, False, False]])
>>> np.argmax(array, axis=0) # evaluate columns
array([0, 0, 3, 2, 1, 0])
>>> np.argmax(array, axis=1) # evaluate rows
array([0, 4, 3, 2])

```
</Transform>
<!--
Notice that we get the same result for column 0 and column 1 as all False returns 0
-->

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
# `np.nonzero()` 1D
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
---
# `np.nonzero()` 2D
<Transform :scale="1.5">

```python {all|1|1-6|7-8|9-10}
>>> array = np.arange(24).reshape(4,6) % 5 == 0
>>> array
array([[ True, False, False, False, False,  True],
       [False, False, False, False,  True, False],
       [False, False, False,  True, False, False],
       [False, False,  True, False, False, False]])
>>> np.nonzero(array) # we get an array per dimension
(array([0, 0, 1, 2, 3]), array([0, 5, 4, 3, 2]))
>>> [array[x, y] for x, y in zip(*np.nonzero(array))]
[True, True, True, True, True]
```
</Transform>
<!--
Notice that we have read through these coordinates to discover the first true per axis
-->


---
layout: center
---
# Performance of `np.argmax()` and `np.nonzero()`


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

Can operate directly on a C array.

Can return a `PyObject` or NumPy array
</v-clicks>
</Transform>


---
---
# Writing Python C-Extensions

<Transform :scale="1.5">
<v-clicks>

Custom types are hard

Single functions are straightforward

Python, NumPy C-APIs are reasonably well documented

Must do cross-platform testing in CI (`cibuildwheel`)
</v-clicks>
</Transform>


---
---
# ``first_true_1d()`` as a C Extension

<Transform :scale="1.5">
<v-clicks depth="2">

- Two Arguments
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


---
---
# I: Reading Native `PyObject`s From Arrays

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


---
---
# I: Reading Native `PyObject`s From Arrays

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


---
---
# II: Reading NumPy Scalar `PyObject`s From Arrays

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


---
---
# II: Reading NumPy Scalar `PyObject`s From Arrays

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


---
---
# III: Casting Data Pointers to C-Types

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


---
---
# III: Casting Data Pointers to C-Types

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

Use `NpyIter_New()` to setup iteration

Generality for N-dimensional arrays of diverse homogeneity

Perform stride-sized pointer arithmetic in inner loop

Requires more code

Does not support reverse iteration
</v-clicks>
</Transform>


---
---
# IV: Using `NpyIter`

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


---
---
# IV: Using `NpyIter`

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


---
---
# IV: Using `NpyIter`

```c {all|1-2|4-5|7-8|10}
    npy_bool **data_ptr_array = (npy_bool**)NpyIter_GetDataPtrArray(iter);
    npy_bool *data_ptr;

    npy_intp *stride_ptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp stride;

    npy_intp *inner_size_ptr = NpyIter_GetInnerLoopSizePtr(iter);
    npy_intp inner_size;

    npy_intp i = 0;
```


---
---
# IV: Using `NpyIter`

```c {all|1,13|2-4|6,12|7-9,15-16|10-11|14}
    do {
        data_ptr = *data_ptr_array;
        stride = *stride_ptr;
        inner_size = *inner_size_ptr;

        while (inner_size--) {
            if (*data_ptr) {
                goto end;
            }
            i++;
            data_ptr += stride;
        }
    } while(iter_next(iter));
    return PyLong_FromSsize_t(-1);
end:
    return PyLong_FromSsize_t(i);
```


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
# V(a): Using C-Arrays and Pointer Arithmetic

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean, and *contiguous* arrays

Use `PyArray_DATA()` to get pointer to underlying C-array

Advance through array with pointer arithmetic
</v-clicks>
</Transform>

---
---
# V(a): Using C-Arrays and Pointer Arithmetic
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

---
---
# V(a): Using C-Arrays and Pointer Arithmetic
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

---
---
# V(a): Using C-Arrays and Pointer Arithmetic
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

Loop unrolling... even with modern compilers

Single instruction, multiple data (SIMD) instructions
</v-clicks>
</Transform>


---
---
# V(b): Using C-Arrays, Pointer Arithmetic, Loop Unrolling

<Transform :scale="1.5">
<v-clicks>

Unrolling is a space versus time tradeoff

Only process 1D, Boolean, contiguous arrays

Use `PyArray_DATA()` to get C-array

Advance through array with pointer arithmetic, unrolling units of 4
</v-clicks>
</Transform>


---
---
# V(b): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
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


---
---
# V(b): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
```c {all|1|3-4|6-9}
    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    npy_intp size = PyArray_SIZE(array);
    lldiv_t size_div = lldiv((long long)size, 4);

    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;
```


---
---
# V(b): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
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


---
---
# V(b): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
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


---
---
# V(b): Using C-Arrays, Pointer Arithmetic, Loop Unrolling
```c {all}
    if (p != p_end) {
        position = p - array_buffer;
    }
    return PyLong_FromSsize_t(position);
```


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
    - Use pointer arithmetic (and loop unrolling) through each row
    - Short-circuit if `True` found, jump to next row
- F-contiguous and Axis 0 works the same
- Use `PyArray_NewCopy()` to get new contiguous ordering
- Use `PyArray_DATA()` to get C-array

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

Sli.dev slides

Code: https://github.com/flexatone/np-bench

`first_true_1d`, `first_true_2d` packaged: https://pypi.org/project/arraykit

StaticFrame: https://static-frame.dev

</Transform>
