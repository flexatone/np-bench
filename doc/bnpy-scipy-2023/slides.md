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

Python (using C PyObjects) is relatively slow

C-extensions offer opportunities for using C-types (at C performance)

With NumPy, we get flexible usage of C-arrays in Python

</v-clicks>
</Transform>


---
---
# My Journey

<Transform :scale="1.5">
<v-clicks>

Built StaticFrame leveraging NumPy

Performance studies identify opportunities

For pure Python, can implement routines in C-types

What about routines that are already using NumPy?
</v-clicks>
</Transform>


---
---
# Can NumPy Routines be Optimized?

<Transform :scale="1.5">
<v-clicks>

Some NumPy routines are implemented in Python

Some NumPy routines might do more than we need

</v-clicks>
</Transform>


---
---
# Can NumPy Routines be Optimized?

<Transform :scale="1.5">
<v-clicks>

Many NumPy routines are flexible

* Handle N-dimensional arrays

* Handle full diversity of dtypes

* Handle non-array (i.e., list, tuple) inputs


More narrow routines might be able to out-perform flexible routines

</v-clicks>
</Transform>

<style>
ul li {list-style-type: disc;}
</style>



---
---
# Case Study: Finding the First True in an Array

<Transform :scale="1.5">
<v-clicks>

Given a 1D Boolean array, what is the index of the first `True`

Given a 2D Boolean array, what are the indices of the first True per axis

Need to be able to search in both directions

Need to know if there are no `True`
</v-clicks>
</Transform>



---
layout: none
---
# Stack Overflow 1

<div class="absolute top-0px">
<img src="/screen-so-1.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>


---
layout: none
---
# Stack Overflow 2

<div class="absolute top-0px">
<img src="/screen-so-2.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>


---
layout: none
---
# NumPy Issue 2269

<div class="absolute top-0px">
<img src="/screen-gh-npy-issue-2269.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>


---
---
# Finding the First True: NumPy Options

<Transform :scale="1.5">

`np.argmax()`

`np.nonzero()`


</Transform>


---
---
# `np.argmax()` 1D
<Transform :scale="1.6">

```python {all|1|2-3|4|5-6|all}
>>> array = np.arange(10_000) == 2_000
>>> np.argmax(array) # finds first True
2000
>>> array = np.full(10_000, False)
>>> np.argmax(array) # if all False, reports 0
0
```
</Transform>

<!--
Cannot distguish all False from True at 0
Finds first value; but cannot get first value from opposite direction
-->


---
---
# `np.argmax()` 2D
<Transform :scale="1.6">

```python {all|1|2-6|7-8|9-10|all}
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
# `np.nonzero()` 1D
<Transform :scale="1.6">

```python {all|1|2-3|4-5|6|7-8|9-10|all} {lines:false}
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
<Transform :scale="1.6">

```python {all|1|2-6|7-8|9-10|all}
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
Notice that we have read through these coordinates to discuver the first true per axis
-->


---
---
# Opportunities for Improvement

<Transform :scale="1.25">
<v-clicks>

- `np.argmax`:

    - Does not handle all-`False` case

    - Could add an `np.any()` call to find all-`False`

    - $\mathcal{O}(2n)$ worst case, but can short-circuit

- `np.nonzero`

    - Cannot short-circuit

    - Always $\mathcal{O}(n)$ as cannot short-circuit

- Both options are suboptimal

</v-clicks>
</Transform>

<!-- What if we write a C-extension to do just what we need

Only handle 1D, 2D contiguous Boolean arrays

Return -1 when all `False` -->


---
---
# Many Options for Performance

<Transform :scale="1.5">
<v-clicks>

C-Extensions

Cython

Numba

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

A path for doing what needs to be done without PyObjects

Can use an input array (or arrays) as a C array.

Can build a C array (and return PyObject array)
</v-clicks>
</Transform>


---
---
# Writing Python C-Extensions

<Transform :scale="1.5">
<v-clicks>

Custom types are hard

Writing single functions is not that bad

Python, NumPy C-APIs are reasonably well documented

Must do cross-platform testing in CI (`cibuildwheel`)
</v-clicks>
</Transform>



---
---
# ``first_true_1d()`` in C

<Transform :scale="1.5">
<v-clicks>

Take an array and a forward Boolean (where False is reverse)

Evaluate elements, return the index of the first `True`

If no `True`, return -1

All Code: https://github.com/flexatone/np-bench

</v-clicks>
</Transform>


---
---
# A Minimal C Extension

```c
// excluding define, include statements
static struct PyModuleDef npb_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "np_bench",
    .m_size = -1,
};

PyObject *
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

```c
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
    PyObject* post = PyLong_FromSsize_t(-1); // temporarily always return -1
    return post;
}
```

---
---
# Adding a C Function to a Python Module

```c
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
---
# Five Ways to Read 1D Array Data

<Transform :scale="1.5">
<v-clicks>

```c
PyArray_GETITEM(array, PyArray_GETPTR1(array, i))
```

```c
PyArray_ToScalar(PyArray_GETPTR1(array, i), array)
```

```c
PyArray_GETPTR1(array, i)
```

```c
NpyIter_New()
```

```c
PyArray_DATA(array)
```
</v-clicks>
</Transform>



---
---
# I: Reading Native `PyObject`s From Arrays

<Transform :scale="1.5">
<v-clicks>

Only process 1D arrays

Use `PyArray_GETPTR1()`, then `PyArray_GETITEM()`

Convert array element to `PyObject`

Use Python C-API `PyObject_IsTrue()` to evaluate elements

Must manage reference counting for `PyObject`s
</v-clicks>
</Transform>


---
---
# I: Reading Native `PyObject`s From Arrays

```c
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
    ...
```


---
---
# I: Reading Native `PyObject`s From Arrays

```c
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

```c
    else { // not forward
        for (i = size - 1; i >= 0; i--) {
            element = PyArray_GETITEM(array, PyArray_GETPTR1(array, i));
            if(PyObject_IsTrue(element)) {
                Py_DECREF(element);
                break;
            }
            Py_DECREF(element);
        }
    }
    if (i < 0 || i >= size ) { // else, return -1
        i = -1;
    }
    return PyLong_FromSsize_t(i);
}
```



---
layout: none
---
# I: Reading Native `PyObject`s From Arrays

<div class="absolute top-0px">
<img src="" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>



---
---
# II: Reading Scalar `PyObject`s From Arrays

<Transform :scale="1.5">
<v-clicks>

Only process 1D arrays

Use `PyArray_GETPTR1()`, then `PyArray_ToScalar()`

Array scalars are `PyObject`s

Use Python C-API `PyObject_IsTrue()` to evaluate elements

Must manage reference counting for `PyObject`s
</v-clicks>
</Transform>



---
layout: none
---
# II: Reading Scalar `PyObject`s From Arrays

<div class="absolute top-0px">
<img src="" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>





---
---
# III: Casting Data Pointers to C-Types

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean arrays

Use `PyArray_GETPTR1()` and cast to C type

No use of Python C-API, no reference counting
</v-clicks>
</Transform>



---
layout: none
---
# III: Casting Data Pointers to C-Types

<div class="absolute top-0px">
<img src="" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>




---
---
# IV: Using `NpyIter`

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean arrays

Use `NpyIter_New()` to setup iteration

Generality for N-dimensional arrays of diverse homogeniety
</v-clicks>
</Transform>



---
layout: none
---
# IV: Using `NpyIter`

<div class="absolute top-0px">
<img src="" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>




---
---
# V(a): Using C-Array and Pointer Arithmetic

<Transform :scale="1.5">
<v-clicks>

Only process 1D, Boolean, contiguous arrays

Use `PyArray_DATA()` to get C-array

Advance through array with pointer arithmetic
</v-clicks>
</Transform>




---
layout: none
---
# V(a): Using C-Array and Pointer Arithmetic

<div class="absolute top-0px">
<img src="" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>





---
---
# V(b): Using C-Array, Pointer Arithmetic, Loop Unrolling

<Transform :scale="1.5">
<v-clicks>

Space-time tradeoff

Only process 1D, Boolean, contiguous arrays

Use `PyArray_DATA()` to get C-array

Advance through array with pointer arithmetic, unrolling units of 4
</v-clicks>
</Transform>



---
layout: none
---
# V(b): Using C-Array, Pointer Arithmetic, Loop Unrolling

<div class="absolute top-0px">
<img src="" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>





---
---
# VI: 2D Forced Contiguous C-Order-Array, Pointer Arithmetic, Loop Unrolling

<Transform :scale="1.5">
<v-clicks>
Only process 1D, Boolean, contiguous arrays

Use `PyArray_DATA()` to get C-array

Advance through array with pointer arithmetic, unrolling units of 4
</v-clicks>
</Transform>



---
layout: none
---
# VI: 2D Forced Contiguous C-Order-Array, Pointer Arithmetic, Loop Unrolling

<div class="absolute top-0px">
<img src="" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>




---
layout: quote
---
# Thanks!

<Transform :scale="1.5">
Thanks to Brandt Bucher & Charles Burkland

Sli.dev slides

Code: https://github.com/flexatone/np-bench

StaticFrame: https://static-frame.dev

</Transform>
