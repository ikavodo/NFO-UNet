import os


def available_cpus() -> int:
    # os.cpu_count() reports the node's total CPUs, not the SLURM job's --cpus-per-task
    # allocation (that's a cgroup/cpuset affinity limit os.cpu_count() doesn't read) -
    # prefer SLURM's own env var when present so num_workers doesn't oversubscribe.
    return int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))


class AbstractConfig(object):

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict=None):
        if new_config_dict is None:
            new_config_dict = {}
        ret = AbstractConfig(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        if isinstance(new_config_dict, AbstractConfig):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def __str__(self):
        data = '\nConfig {\n'
        for k, v in vars(self).items():
            v_str = f'{v}'.replace(' ', '').replace('\n', ' ')
            print_msg = f'{k} = {v_str}'
            data = f'{data}  {print_msg}\n'
        return f'{data}{"}"}'
