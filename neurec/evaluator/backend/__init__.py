# import eval_score_matrix_foldout
try:
    from neurec.evaluator.backend.cpp.evaluate_foldout import eval_score_matrix_foldout
    print("eval_score_matrix_foldout with cpp")
except:
    from neurec.evaluator.backend.python.evaluate_foldout import eval_score_matrix_foldout
    print("eval_score_matrix_foldout with python")

# import eval_score_matrix_loo
try:
    from neurec.evaluator.backend.cpp.evaluate_loo import eval_score_matrix_loo
    print("eval_score_matrix_loo with cpp")
except:
    from neurec.evaluator.backend.python.evaluate_loo import eval_score_matrix_loo
    print("eval_score_matrix_loo with python")
