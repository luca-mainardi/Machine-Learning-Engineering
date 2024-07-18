import string

import inspect





def is_legal_answer(q: str, n_options: int = 4):

    assert isinstance(q, str), f"The answer should be a string, was {type(q)}."

    q = q.replace(" ", "").replace(",", "")

    for c in q:

        assert c.lower() in string.ascii_lowercase[:n_options], f"Invalid character in answer: '{c}'"





def _format_parameters(fn):

    def _format_parameter(param):

        return param.name + (f"={param.default}" if not param.default == inspect._empty else "")



    return "(" + ", ".join(_format_parameter(p) for p in inspect.signature(fn).parameters.values()) + ")"





def _check_signature(expected, provided):

    assert str(inspect.signature(provided)) == str(inspect.signature(

        expected)), f"Function signature modified! Expected {_format_parameters(expected)} but got {_format_parameters(provided)}."





def signature_unchanged(fn, *args, **kwargs):
    if fn.__name__ == "evaluate_LR":
        def evaluate_LR(X, y, C):
            """ Evaluate a Logistic Regression model with cross-validation on the provided image data.
            Keyword arguments:
            X -- the data for training and testing
            y -- the correct output values
            C -- the regularization constant

            Returns: a dictionary with the mean train and test score, e.g. {"train": 0.9, "test": 0.95}
            """
            pass
        _check_signature(evaluate_LR, fn)
    if fn.__name__ == "plot_curve":
        def plot_curve(X,y,train_size):
            """ Plots the train and test accuracy of logistic regression on a
            subsample of the given data for different amounts of regularization.
            X          -- the data for training and testing
            y          -- the correct labels
            train_size -- the proportion of the data used for training and testing, between 0.0 and 1.0. 
    
            Returns: a plot as described above, with C on the x-axis and accuracy on
            the y-axis.
            """
            pass
        _check_signature(plot_curve, fn)
    if fn.__name__ == "plot_curve_embedded":
        def plot_curve_embedded(X,y,train_size):
            """ Plots the train and test accuracy of logistic regression on a
            subsample of the given data for different amounts of regularization.
            X -- the data for training and testing
            y -- the correct labels
            train_size -- the proportion of the data used for training and testing, between 0.0 and 1.0. 

            Returns: a plot as described above, with C on the x-axis and accuracy on
            the y-axis.
            """
            pass
        _check_signature(plot_curve_embedded, fn)
    if fn.__name__ == "evaluate_pixel":
        def evaluate_pixel(X_train, y_train, X_eval, y_eval):
            """ Evaluate a Logistic Regression model
            X_train -- the training data
            y_train -- the training labels
            X_eval  -- the evaluation (test) data
            y_eval  -- the evaluation (test) labels
    
            Returns: the evaluation score (accuracy) of the optimal model trained on pixel data
            """
            pass
        _check_signature(evaluate_pixel, fn)
    if fn.__name__ == "evaluate_embedding":
        def evaluate_embedding(X_train, y_train, X_eval, y_eval):
            """ Evaluate a Logistic Regression model
            X_train -- the training data
            y_train -- the training labels
            X_eval  -- the evaluation (test) data
            y_eval  -- the evaluation (test) labels
    
            Returns: the evaluation score (accuracy) of the optimal model trained on pixel data
            """
            pass
        _check_signature(evaluate_embedding, fn)
    if fn.__name__ == "plot_character_coefficients":
        def plot_character_coefficients(X, y, character):
            """ Plots 32x32 heatmaps showing the coefficients of three Logistic
            Regression models, each with different amounts of regularization values.
            X -- the data for training and testing
            y -- the correct labels

            Returns: 4 plots, as described above.
            """
            pass
        _check_signature(plot_character_coefficients, fn)
    if fn.__name__ == "plot_confusion_matrix":
        def plot_confusion_matrix(X, y):
          pass
        _check_signature(plot_confusion_matrix, fn)
