NUM_CLUSTERS = 3
TABNET_EPOCHS = 50
STACK_CV      = 3
THRESHOLD_FN  = 0.10   # доля FN от всех ошибок

CAT_PARAMS = dict(depth=6, learning_rate=0.1, iterations=300)
LGB_PARAMS = dict(num_leaves=31, learning_rate=0.05, n_estimators=300)
XGB_PARAMS = dict(max_depth=6, eta=0.1, n_estimators=300)