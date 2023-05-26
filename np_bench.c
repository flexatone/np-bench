# include <math.h>
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

static struct PyModuleDef npb_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_doc = "NumPy Performance Benchmarks",
    .m_name = "arraymap",
    .m_size = -1,
};

PyObject *
PyInit_np_bench(void)
{
    import_array();
    PyObject *m = PyModule_Create(&npb_module);
    if (
        !m
        || PyModule_AddStringConstant(m, "__version__", Py_STRINGIFY(AM_VERSION))
    ) {
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}

