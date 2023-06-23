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



