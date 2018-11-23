import numpy as np

# create a function that gives softmax probability of NxD dimension input x
# It must include max subtraction of max(xi) row
# The output dimension should be of NXD

def softmax(x):
    # columnwise max for matrix x, rowise max for vector x
    shape = x.shape
    if len(shape) == 1:
        max = np.max(x, axis=0)
    elif len(shape) == 2:
        N = shape[0]
        max = np.max(x, axis=1)
        max = np.tile(max, (N,1))
        max = max.T
    else:
        raise Exception

    # subtract max
    x = np.subtract(x, max)

    # get sum over each col
    x = np.exp(x)
    if len(shape) == 1:
        sum = np.sum(x, axis=0) # scalar
    elif len(shape) == 2:
        sum = np.sum(x, axis=1) # Nx1
    else:
        raise Exception
    
    # divide each row with sum
    x = np.divide(x, sum)
    return x

def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print( "You should be able to verify these results by hand!\n")


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    test_softmax_basic()