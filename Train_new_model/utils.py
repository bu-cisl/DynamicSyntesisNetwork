

def get_all_variables_from_top_scope(tf, scope):
    #scope is a top scope here, otherwise change startswith part
    return [v for v in tf.all_variables() if v.name.startswith(scope.name)]