from statistics import fmean, stdev
class logger:
    def __init__(self, **kwargs) -> None:
        '''
        BestSol:        Save the evolution of the best solution so far.
        BestObj:        Save the evolution of the objective value of the best solution so far.
        Pop:            Save the evolution of the population.
        IterBestSol:    Save the evolution of the best solution in each iteration.
        IterBestObj:    Save the evolution of the objective value of the best solution in each iteration.
        IterObjAvg:     Save the evolution of the averaged objective value in each iteration.
        IterObjSTD:     Save the evolution of the standard deviation of objective values in each iteration.
        '''
        self.BestSol = kwargs.get("BestSol", True)
        self.BestObj = kwargs.get("BestObj", True)
        self.Pop = kwargs.get("Pop", False)
        self.IterBestSol = kwargs.get("IterBestSol", False)
        self.IterBestObj = kwargs.get("IterBestObj", False)
        self.IterObjAvg = kwargs.get("IterObjAvg", True)
        self.IterObjSTD = kwargs.get("IterObjSTD", True)
        
    def Log(self, Copy, BestSol, BestObj, IterBestSol, IterBestObj, Population, PopObjs):
        log = []
        logstr = ""
        if self.BestSol:
            log.append(Copy(BestSol))
        if self.BestObj:
            log.append(BestObj)
            logstr += f"Best: {BestObj:.4f}    "
        if self.Pop:
            log.append([Copy(p) for p in Population])
        if self.IterBestSol:
            log.append(Copy(IterBestSol))
        if self.IterBestObj:
            log.append(IterBestObj)
            logstr += f"IterBest: {IterBestObj:.4f}    "
        if self.IterObjAvg:
            log.append(fmean(PopObjs))
            logstr += f"IterAvg: {log[-1]:.4f}    "
        if self.IterObjSTD:
            log.append(stdev(PopObjs))
            logstr += f"IterSTD: {log[-1]:.4f}    "

        return log, logstr