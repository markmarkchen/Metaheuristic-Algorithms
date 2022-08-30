import numpy as np
import pickle
import datetime, time

class Pipeline:
    def __init__(self, Problem, Populations = [], Solver = [], Repeat = 3):
        self.Solver = Solver
        self.Populations = Populations
        self.Repeat = Repeat
        self.Problem = Problem
        self.Log = []
    def Reset(self):
        self.Log = []

    def Run(self):
        Bounds = self.Problem.VariableBounds
        F = self.Problem.ObjectiveFunction
        R = self.Problem.Repair
        E = self.Problem.FullObjFunction

        for i, pop in enumerate(self.Populations):
            log = []
            for solver in self.Solver:
                solver.VariableBounds = Bounds
                solver.ObjectiveFunction = F
                solver.ExplicitObjFunc = E
                solver.Repair = R
                slog = [type(solver).__name__]
                
                for repear in range(self.Repeat):
                    solver.Reset()
                    solver.Population = pop.copy()
                    self.Problem.NumObjCall = 0                    
                    solver.Run()
                    slog.append([[i for i in j] for j in solver.Log])

                log.append(slog)
            
            self.Log.append(log)

    def Save(self):
        with open(f"{type(self.Problem).__name__} {datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')} Log.pickle", "wb") as f:
            pickle.dump(self.Log, f)

                            