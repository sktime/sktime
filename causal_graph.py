
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel

def learn_dag(data):
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=BicScore(data))
    return model
