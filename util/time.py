class CheckTime(object):
    def __init__(self):
        self.time = datetime.now()

    def finishTime(self):
        delta = datetime.now()-self.time
        self.time = datetime.now()
        return delta
    def printDelta(self,task):
        delta = self.finishTime()
        print(task+" done in "+str(delta.total_seconds()))
