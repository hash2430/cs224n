import numpy as np
from tempfile import TemporaryFile
if __name__ == "__main__":
    # Creating Arrays
    a = np.array([1, 2, 3])
    b = np.array([[1.5, 2, 3], [4,5,6]], dtype=float)
    c = np.array([[(1.5, 2, 3), (4, 5, 6)],[(3,2,1),(4,5,6)]],dtype=float)

    # Initial Placeholders
    np.zeros((3,4))
    np.ones((2,3,4), dtype=np.int16)
    d = np.arange(10, 25, 5) #  first, last, interval
    d = np.linspace(0,2,9) # fisrt, last, num_samples
    e = np.full((2,2), 7)
    f = np.eye(2) # Identity matrix
    f = np.random.random((2,2))
    f = np.empty((3, 2))

    # I/O
    ## npz
    outfile = TemporaryFile()
    np.savez(outfile, a, b, c, d, e, f)
    outfile.seek(0)

    npzfile = np.load(outfile)
    print(npzfile.files)
    print(npzfile['arr_0'])

    ## npz with index
    outfile = TemporaryFile()
    np.savez(outfile, a=a, b=b, c=c)
    outfile.seek(0)

    npzfile = np.load(outfile)
    print(npzfile.files)
    print(npzfile['a'])

    ## save text
    np.savetxt("myarray.txt", a, delimiter=" ")
    txt = np.loadtxt("myarray.txt")
    arr = np.genfromtxt("myarray.txt", delimiter=',')
    print(txt)

    # Inspecting your array
    print(c.shape)
    print(len(c))
    print(c.ndim)
    print(c.size)
    print(c.dtype)
    print(c.dtype.name)
    print(c.astype(int))

    # Asking for help
    print(np.info(np.ndarray.dtype))
    print()

    # Array Mathmatics
    ## Arithmetic Operations
    print(np.multiply(a,b))
    x = np.array([1,2,3])
    y = np.array([[1],[2],[3]])
    print(np.matmul(x,y))
    print(y.dot(x.reshape(1,3)))

    # Comparison
    print("a == b")
    print(a == b)
    print("a < b")
    print(a < b)
    print(np.array_equal(a,b))
    # Aggregate Function
    print("=== Aggregation ===")
    print("a.sum(): {}".format(a.sum()))
    print("a.min(): {}".format(a.min()))
    print("a.max(): {}".format(a.max()))
    print("a.cumsum(): {}".format(a.cumsum()))
    print("a.mean(): {}".format(a.mean()))
    print("np.median(a): {}".format(np.median(a)))
    print("np.corrcoef(a): {}".format(np.corrcoef(a)))
    print("np.std(a): {}".format(np.std(a)))
    # Copying Arrays
    print("view")
    a = np.array([1, 2, 3])
    h = a.view()
    a[0] = 10
    print(h)

    print("deep copy")
    a = np.array([1, 2, 3])
    h = a.copy()
    a[0] = 10
    print(h)

    print("copy")
    a = np.array([1, 2, 3])
    h = np.copy(a)
    a[0]=10
    print(h)

    ## Indexing and slicing returns a view
    print("Indexing and slicing")
    a = np.array([1, 2, 3])
    h = a[0:2]
    h[0] = 10
    print(a)

    ## Indexing and slicing then copying returns a copy
    print("Indexing and slicing then copy")
    a = np.array([1, 2, 3])
    h = a[0:2].copy()
    h = 1
    print(a)
    ## Fancy Indexing returns a copy
    print("Fancy indexing")
    a = np.array([1,2,3])
    h = a[[-3, -2, -1]]
    h[0] = 10
    print(h)
    print(a)

    ## Fancy indexing 2D
    a = np.array([[1,2,3],[4,5,6]])
    h = a[[0,1]][:,[0]]
    h[0] = 10
    print(h)
    print(a)


    # Sorting Arrays
    ## numpy.ndarray.sort() has no return
    print("=== Sort ===")
    a = np.array([[1,4],[3,1]])
    b=a.sort(axis=1)
    print(a)
    a.sort(axis=0)
    print(a)

    # Array Manipulation
    ## Transpose
    print(a.T.shape)
    ## Changing Array shape
    b=a.ravel()
    print(b)
    b=a.reshape(4,-1)
    print(b)
    a = np.array([1,2,3,4])
    a.resize((5,3))
    print(a)
    a = np.array([1,2,3,4])
    b = np.array([5,6,7,8])
    c = np.append(a,b)
    print(c)
    c = np.insert(a, 1, 5)
    print(c)
    a = np.array([1, 2, 3, 4])
    c = np.delete(a, 0)
    print(c)