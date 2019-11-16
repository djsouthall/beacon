import cProfile, pstats, io

def profile(fnc):
    '''
    A decorator that uses cProfile to profile a function
    This is lifted from https://osf.io/upav8/
    
    Required imports:
    import cProfile, pstats, io
    
    To use, decorate function of interest by putting @gnosim.utils.misc.profile above
    its definition.
    
    Meanings:
    ncalls  - for the number of calls.  When there are two numbers (for example 3/1), 
              it means that the function recursed. The second value is the number 
              of primitive calls and the former is the total number of calls. Note 
              that when the function does not recurse, these two values are the same, 
              and only the single figure is printed.
    tottime - for the total time spent in the given function (and excluding time made 
              in calls to sub-functions)
    percall - is the quotient of tottime divided by ncalls
    cumtime - is the cumulative time spent in this and all subfunctions (from invocation 
              till exit). This figure is accurate even for recursive functions.
    percall - is the quotient of cumtime divided by primitive calls
    filename:lineno(function) - provides the respective data of each function
    '''
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs().sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner