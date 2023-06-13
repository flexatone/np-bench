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
# We Can Do Better... Right?

<Transform :scale="1.5">
<v-clicks>

`np.argmax` suffers from not handling all-`False` case

`np.nonzero` must do a full collection and cannot short circuit

What if we write a C-extension to do just what we need

Only handle 1D, 2D contiguous Boolean arrays

Return -1 when all `False`
</v-clicks>
</Transform>



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
# A C Function as Module-Level Python Function

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
    PyObject* post = PyLong_FromSsize_t(-1);
    return post;
}
```

---
---
# Adding a Function to a Module

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
# Many Ways to Read Array Data

<Transform :scale="1.5">
<v-clicks>

Use `PyArray_GETPTR1()`

Use `NpyIter`

Use `PyArray_DATA` and pointer arithmatic
</v-clicks>
</Transform>

