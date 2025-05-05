# Fibonacci numbers module

def fib(n: object) -> object:    # write Fibonacci series up to n
    """

    Parameters
    ----------
    n
    """
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()

def fib2(n: object) -> object:   # return Fibonacci series up to n
    """

    Parameters
    ----------
    n

    Returns
    -------

    """
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result

''' Special construct that let us run the module as a standalone script.'''
if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))