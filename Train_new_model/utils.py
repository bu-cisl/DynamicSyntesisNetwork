

# return all variables from the top scope
def get_all_variables_from_top_scope(tf, scope):
    return [v for v in tf.all_variables() if v.name.startswith(scope.name)]