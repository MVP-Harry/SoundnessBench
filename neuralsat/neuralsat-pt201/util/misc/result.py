from collections import namedtuple


AbstractResults = namedtuple(
    'AbstractResults', 
    (
        'objective_ids',
        'output_lbs', 
        'masks', 'lAs', 'histories', 
        'lower_bounds', 'upper_bounds', 
        'input_lowers', 'input_uppers',
        'slopes', 'betas', 
        'cs', 'rhs',
        'sat_solvers'
    ), 
    defaults=(None,) * 14
)


CoefficientMatrix = namedtuple(
    'CoefficientMatrix', ('lA', 'uA', 'lbias', 'ubias'), 
    defaults=(None,) * 4
)

class ReturnStatus:

    UNSAT       = 'unsat'
    SAT         = 'sat'
    UNKNOWN     = 'unknown'
    TIMEOUT     = 'timeout'
    RESTART     = 'restart'
    INVALID_CEX = 'invalid_counterexample'
