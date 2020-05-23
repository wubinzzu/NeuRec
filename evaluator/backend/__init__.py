try:
    from evaluator.backend.cpp.uni_evaluator import UniEvaluator
    print("Evaluate model with cpp")
except:
    from evaluator.backend.python.uni_evaluator import UniEvaluator
    print("Evaluate model with python")
