import numpy as np
import random
# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    # x의 각각의 dimension에 대한 각각의 gradient가 numerical gradient와 일치하는지 dimension 별로 체크해봐야함
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # x[ix] is writable (by its op_flags) thus it needs to be restored at the end of each turn
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        """
        specify ix index for the movement of x in only one dimension and not the others
        x is changed 
        """
        x[ix] = x[ix] + h
        random.setstate(rndstate)
        right_fx,_ = f(x)
        x[ix] = x[ix] - 2 * h
        random.setstate(rndstate)
        left_fx, _ = f(x)
        x[ix] = x[ix] + h
        numeric_grad = (right_fx - left_fx)/(2*h)
        # normalization_term: gradient 자체의 크기에 따라 허용되는 gradient 오차 범위도 달라지기 대문에 normalization 필요
        normalization_term = max(1, numeric_grad, grad[ix])
        grad_error = np.abs(numeric_grad - grad[ix]) / normalization_term
        assert(grad_error < h)
        it.iternext()
    print("Gradient check passed!")

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print( "Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print("")


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()


