class AbstractRecommender(object):
    def __init__(self):  
        
        raise NotImplementedError
    def build_graph(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError
    
    def evaluate_model(self): 
        raise NotImplementedError 