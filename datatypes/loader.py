import yaml

from datatypes.sets import ParameterSet, InitialSet, IntervalSet
def get_params(param_path, equilibrium):
    with open(param_path, "r") as f:
        config = yaml.safe_load(f)
    return ParameterSet(**config[equilibrium]), InitialSet(**config['initial']), IntervalSet(**config['interval'])