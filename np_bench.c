# include <math.h>
# include "stdbool.h"

# define PY_SSIZE_T_CLEAN
# include "Python.h"

# define PY_ARRAY_UNIQUE_SYMBOL NPB_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"
# include "numpy/halffloat.h"

# define DEBUG_MSG_OBJ(msg, obj)      \
    fprintf(stderr, "--- %s: %i: %s: ", __FILE__, __LINE__, __FUNCTION__); \
    fprintf(stderr, #msg " ");        \
    PyObject_Print(obj, stderr, 0);   \
    fprintf(stderr, "\n");            \
    fflush(stderr);                   \



//------------------------------------------------------------------------------
// NOTE: forward determines search priority, either from left or right; indices are always returned relative to the start of the axis.

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
    else {
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
}


static PyObject*
first_true_1d_scalar(PyObject *Py_UNUSED(m), PyObject *args)
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
}





static PyObject*
first_true_1d_getptr(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArrayObject *array = NULL;
    int forward = 1;

    if (!PyArg_ParseTuple(args,
            "O!p:first_true_1d_getptr",
            &PyArray_Type, &array,
            &forward)) {
        return NULL;
    }

    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }

    npy_intp size = PyArray_SIZE(array);
    npy_intp i;

    if (forward) {
        for (i = 0; i < size; i++) {
            if(*(npy_bool*)PyArray_GETPTR1(array, i)) {
                break;
            }
        }
    }
    else {
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
}


static PyObject*
first_true_1d_npyiter(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArrayObject *array = NULL;
    int forward = 1;

    if (!PyArg_ParseTuple(args,
            "O!p:first_true_1d_npyiter",
            &PyArray_Type, &array,
            &forward)) {
        return NULL;
    }

    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }

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

    npy_bool **data_ptr_array = (npy_bool**)NpyIter_GetDataPtrArray(iter);
    npy_bool *data_ptr;

    npy_intp *stride_ptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp stride;

    npy_intp *inner_size_ptr = NpyIter_GetInnerLoopSizePtr(iter);
    npy_intp inner_size;

    npy_intp i = 0;

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
    NpyIter_Deallocate(iter);
    return PyLong_FromSsize_t(-1);
end:
    NpyIter_Deallocate(iter);
    return PyLong_FromSsize_t(i);
}


static PyObject*
first_true_1d_ptr(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArrayObject *array = NULL;
    int forward = 1;

    if (!PyArg_ParseTuple(args,
            "O!p:first_true_1d_ptr",
            &PyArray_Type, &array,
            &forward)) {
        return NULL;
    }

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

    npy_intp size = PyArray_SIZE(array);
    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;

    if (forward) {
        p = array_buffer;
        p_end = p + size;
        while (p < p_end) {
            if (*p) break;
            p++;
        }
    }
    else {
        p = array_buffer + size - 1;
        p_end = array_buffer - 1;
        while (p > p_end) {
            if (*p) break;
            p--;
        }
    }
    if (p != p_end) {
        position = p - array_buffer;
    }

    return PyLong_FromSsize_t(position);
}


static PyObject*
first_true_1d_ptr_unroll(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArrayObject *array = NULL;
    int forward = 1;

    if (!PyArg_ParseTuple(args,
            "O!p:first_true_1d_ptr_unroll",
            &PyArray_Type, &array,
            &forward)) {
        return NULL;
    }

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

    npy_intp size = PyArray_SIZE(array);
    lldiv_t size_div = lldiv((long long)size, 4); // quot, rem

    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;

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
    else {
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
    if (p != p_end) {
        position = p - array_buffer;
    }
    NPY_END_THREADS;

    return PyLong_FromSsize_t(position);
}

#define MEMCMP_SIZE 8
static PyObject*
first_true_1d_memcmp(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArrayObject *array = NULL;
    int forward = 1;

    if (!PyArg_ParseTuple(args,
            "O!p:first_true_1d_memcmp",
            &PyArray_Type, &array,
            &forward)) {
        return NULL;
    }

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

    static npy_bool zero_buffer[MEMCMP_SIZE] = {0};

    npy_intp size = PyArray_SIZE(array);
    lldiv_t size_div = lldiv((long long)size, MEMCMP_SIZE); // quot, rem

    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;

    if (forward) {
        p = array_buffer;
        p_end = p + size;

        while (p < p_end - size_div.rem) {
            if (memcmp(p, zero_buffer, MEMCMP_SIZE) != 0) {
                break;
            } // found a true
            p += MEMCMP_SIZE;
        }
        while (p < p_end) {
            if (*p) {break;}
            p++;
        }
    }
    else {
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
    NPY_END_THREADS;

    return PyLong_FromSsize_t(position);
}


static char *first_true_2d_kwarg_names[] = {
    "array",
    "forward",
    "axis",
    NULL
};

static PyObject*
first_true_2d_unroll(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *array = NULL;
    int forward = 1;
    int axis = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O!|$pi:first_true_2d_unroll",
            first_true_2d_kwarg_names,
            &PyArray_Type,
            &array,
            &forward,
            &axis
            )) {
        return NULL;
    }
    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be 2-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    if (axis < 0 || axis > 1) {
        PyErr_SetString(PyExc_ValueError, "Axis must be 0 or 1");
        return NULL;
    }

    // NOTE: we copy the entire array into contiguous memory when necessary.
    // axis = 0 returns the pos per col
    // axis = 1 returns the pos per row (as contiguous bytes)
    // if c contiguous:
    //      axis == 0: transpose, copy to C
    //      axis == 1: keep
    // if f contiguous:
    //      axis == 0: transpose, keep
    //      axis == 1: copy to C
    // else
    //     axis == 0: transpose, copy to C
    //     axis == 1: copy to C

    bool transpose = !axis; // if 1, false
    bool corder = true;
    if ((PyArray_IS_C_CONTIGUOUS(array) && axis == 1) ||
        (PyArray_IS_F_CONTIGUOUS(array) && axis == 0)) {
        corder = false;
    }
    // create pointer to "indicator" array; if newly allocated, it will need to be decrefed before function termination
    PyArrayObject *array_ind = NULL;
    bool decref_array_ind = false;

    if (transpose && !corder) {
        array_ind = (PyArrayObject *)PyArray_Transpose(array, NULL);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else if (!transpose && corder) {
        array_ind = (PyArrayObject *)PyArray_NewCopy(array, NPY_CORDER);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else if (transpose && corder) {
        PyArrayObject *tmp = (PyArrayObject *)PyArray_Transpose(array, NULL);
        if (tmp == NULL) return NULL;

        array_ind = (PyArrayObject *)PyArray_NewCopy(tmp, NPY_CORDER);
        Py_DECREF((PyObject*)tmp);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else {
        array_ind = array; // can use array, no decref needed
    }

    // buffer of indicators
    npy_bool *buffer_ind = (npy_bool*)PyArray_DATA(array_ind);

    npy_intp count_row = PyArray_DIM(array_ind, 0);
    npy_intp count_col = PyArray_DIM(array_ind, 1);

    ldiv_t div_col = ldiv((long)count_col, 4); // quot, rem

    npy_intp dims_post = {count_row};
    PyArrayObject *array_pos = (PyArrayObject*)PyArray_EMPTY(
            1,         // ndim
            &dims_post,// shape
            NPY_INT64, // dtype
            0          // fortran
            );
    if (array_pos == NULL) {
        return NULL;
    }
    npy_int64 *buffer_pos = (npy_int64*)PyArray_DATA(array_pos);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    npy_intp position;
    npy_bool *p;
    npy_bool *p_start;
    npy_bool *p_end;

    // iterate one row at a time; short-circult when found
    // for axis 1 rows are rows; for axis 0, rows are (post transpose) columns
    for (npy_intp r = 0; r < count_row; r++) {
        position = -1; // update for each row

        if (forward) {
            // get start of each row
            p_start = buffer_ind + (count_col * r);
            p = p_start;
            p_end = p + count_col; // end of each row

            // scan each row from the front and terminate when True
            // remove from the end the remainder
            while (p < p_end - div_col.rem) {
                if (*p) break;
                p++;
                if (*p) break;
                p++;
                if (*p) break;
                p++;
                if (*p) break;
                p++;
            }
            while (p < p_end) {
                if (*p) break;
                p++;
            }
            if (p != p_end) {
                position = p - p_start;
            }
        }
        else {
            // start at the next row, then subtract one for last elem in previous row
            p_start = buffer_ind + (count_col * (r + 1)) - 1;
            p = p_start;
            // end is 1 less than start of each row
            p_end = buffer_ind + (count_col * r) - 1;

            while (p > p_end + div_col.rem) {
                if (*p) break;
                p--;
                if (*p) break;
                p--;
                if (*p) break;
                p--;
                if (*p) break;
                p--;
            }
            while (p > p_end) {
                if (*p) break;
                p--;
            }
            if (p != p_end) {
                position = p - (p_end + 1);
            }
        }
        *buffer_pos++ = position;
    }

    NPY_END_THREADS;

    if (decref_array_ind) {
        Py_DECREF(array_ind); // created in this function
    }
    return (PyObject *)array_pos;
}



static PyObject*
first_true_2d_memcmp(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *array = NULL;
    int forward = 1;
    int axis = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O!|$pi:first_true_2d_memcmp",
            first_true_2d_kwarg_names,
            &PyArray_Type,
            &array,
            &forward,
            &axis
            )) {
        return NULL;
    }
    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be 2-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    if (axis < 0 || axis > 1) {
        PyErr_SetString(PyExc_ValueError, "Axis must be 0 or 1");
        return NULL;
    }

    // NOTE: we copy the entire array into contiguous memory when necessary.
    // axis = 0 returns the pos per col
    // axis = 1 returns the pos per row (as contiguous bytes)
    // if c contiguous:
    //      axis == 0: transpose, copy to C
    //      axis == 1: keep
    // if f contiguous:
    //      axis == 0: transpose, keep
    //      axis == 1: copy to C
    // else
    //     axis == 0: transpose, copy to C
    //     axis == 1: copy to C

    bool transpose = !axis; // if 1, false
    bool corder = true;
    if ((PyArray_IS_C_CONTIGUOUS(array) && axis == 1) ||
        (PyArray_IS_F_CONTIGUOUS(array) && axis == 0)) {
        corder = false;
    }
    // create pointer to "indicator" array; if newly allocated, it will need to be decrefed before function termination
    PyArrayObject *array_ind = NULL;
    bool decref_array_ind = false;

    if (transpose && !corder) {
        array_ind = (PyArrayObject *)PyArray_Transpose(array, NULL);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else if (!transpose && corder) {
        array_ind = (PyArrayObject *)PyArray_NewCopy(array, NPY_CORDER);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else if (transpose && corder) {
        PyArrayObject *tmp = (PyArrayObject *)PyArray_Transpose(array, NULL);
        if (tmp == NULL) return NULL;

        array_ind = (PyArrayObject *)PyArray_NewCopy(tmp, NPY_CORDER);
        Py_DECREF((PyObject*)tmp);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else {
        array_ind = array; // can use array, no decref needed
    }

    // buffer of indicators
    npy_bool *buffer_ind = (npy_bool*)PyArray_DATA(array_ind);

    npy_intp count_row = PyArray_DIM(array_ind, 0);
    npy_intp count_col = PyArray_DIM(array_ind, 1);

    static npy_bool zero_buffer[MEMCMP_SIZE] = {0};
    lldiv_t div_col = lldiv((long long)count_col, MEMCMP_SIZE); // quot, rem

    npy_intp dims_post = {count_row};
    PyArrayObject *array_pos = (PyArrayObject*)PyArray_EMPTY(
            1,         // ndim
            &dims_post,// shape
            NPY_INT64, // dtype
            0          // fortran
            );
    if (array_pos == NULL) {
        return NULL;
    }
    npy_int64 *buffer_pos = (npy_int64*)PyArray_DATA(array_pos);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    npy_intp position;
    npy_bool *p;
    npy_bool *p_start;
    npy_bool *p_end;

    // iterate one row at a time; short-circult when found
    // for axis 1 rows are rows; for axis 0, rows are (post transpose) columns
    for (npy_intp r = 0; r < count_row; r++) {
        position = -1; // update for each row

        if (forward) {
            // get start of each row
            p_start = buffer_ind + (count_col * r);
            p = p_start;
            p_end = p + count_col; // end of each row

            // scan each row from the front and terminate when True
            // remove from the end the remainder
            while (p < p_end - div_col.rem) {
                if (memcmp(p, zero_buffer, MEMCMP_SIZE) != 0) {break;} // found a true
                p += MEMCMP_SIZE;
            }
            while (p < p_end) {
                if (*p) break;
                p++;
            }
            if (p != p_end) {
                position = p - p_start;
            }
        }
        else {
            // start at the next row, then subtract one for last elem in previous row
            p_start = buffer_ind + (count_col * (r + 1)) - 1;
            p = p_start;
            // end is 1 less than start of each row
            p_end = buffer_ind + (count_col * r) - 1;

            while (p > p_end + div_col.rem) {
                if (memcmp(p - MEMCMP_SIZE + 1, zero_buffer, MEMCMP_SIZE) != 0) {
                    break;
                }
                p -= MEMCMP_SIZE;
            }
            while (p > p_end) {
                if (*p) break;
                p--;
            }
            if (p != p_end) {
                position = p - (p_end + 1);
            }
        }
        *buffer_pos++ = position;
    }

    NPY_END_THREADS;

    if (decref_array_ind) {
        Py_DECREF(array_ind); // created in this function
    }
    return (PyObject *)array_pos;
}


//------------------------------------------------------------------------------
// module defintiion

static PyMethodDef npb_methods[] =  {
    {"first_true_1d_getitem", (PyCFunction)first_true_1d_getitem, METH_VARARGS, NULL},
    {"first_true_1d_scalar", (PyCFunction)first_true_1d_scalar, METH_VARARGS, NULL},
    {"first_true_1d_npyiter", (PyCFunction)first_true_1d_npyiter, METH_VARARGS, NULL},
    {"first_true_1d_getptr", (PyCFunction)first_true_1d_getptr, METH_VARARGS, NULL},
    {"first_true_1d_ptr", (PyCFunction)first_true_1d_ptr, METH_VARARGS, NULL},
    {"first_true_1d_ptr_unroll", (PyCFunction)first_true_1d_ptr_unroll, METH_VARARGS, NULL},
    {"first_true_1d_memcmp", (PyCFunction)first_true_1d_memcmp, METH_VARARGS, NULL},
    {"first_true_2d_unroll",
            (PyCFunction)first_true_2d_unroll,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"first_true_2d_memcmp",
            (PyCFunction)first_true_2d_memcmp,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {NULL},
};

static struct PyModuleDef npb_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_doc = "NumPy Performance Benchmarks",
    .m_name = "np_bench",
    .m_size = -1,
    .m_methods = npb_methods,
};

PyObject *
PyInit_np_bench(void)
{
    import_array();
    PyObject *m = PyModule_Create(&npb_module);
    if (
        !m
        || PyModule_AddStringConstant(m, "__version__", Py_STRINGIFY(NPB_VERSION))
    ) {
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}

