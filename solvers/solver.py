def get_solvers(config):
    if config['dataset'] == "CIFAR10":
        from solvers.solver_cifar10 import Solver
        solver     = Solver(config)
    elif config['dataset'] == "FashionMNIST":
        from solvers.solver_fashion import Solver
        solver     = Solver(config)
    elif config['dataset'] == "COIL20":
        from solvers.solver_coil20 import Solver
        solver     = Solver(config)
    else:
        print("Unknown solver")
    return solver