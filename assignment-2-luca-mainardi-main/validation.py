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
    if fn.__name__ == "plot_feature_importance":
        def plot_feature_importance(features,importances):
            """ Plot the features and their importances as a bar chart in order of importance.
            Args:
                features: the feature names, ordered by importance
                importances: the feature importances
            """
        _check_signature(plot_feature_importance, fn)
    if fn.__name__ == "pipeline_builder":
        def pipeline_builder(model, transformers=None):
          """ Returns a pipeline that imputes missing values, runs the given transformers, and then runs the given model.
          Keyword arguments:
            model -- any scikit-learn-compatible model
            transformers -- a list of additional scikit-learn-compatible transformers (Optional)
            Returns: an scikit-learn pipeline which preprocesses the data and then runs the classifier
          """
          pass
        _check_signature(pipeline_builder, fn)
    if fn.__name__ == "baseline_model":
        def baseline_model():
            """ Returns a trained pipeline that imputes missing values, selects features, and scales numeric features as described above
            """
            pass
        _check_signature(baseline_model, fn)
    if fn.__name__ == "predict_total_influx":
        def predict_total_influx():
            """ Builds (but doesn't train) the best model to predict `total_influx`.
            Returns: the model/pipeline
            """
            pass
        _check_signature(predict_total_influx, fn)
    if fn.__name__ == "plot_analyze_power_lines":
        def plot_analyze_power_lines(X,Y):
            """
            Plots a heatmap visualizing the correlation between the power lines (in `Y2`) and the other features (in `X2`)
            """
            pass
        _check_signature(plot_analyze_power_lines, fn)
    if fn.__name__ == "predict_power_lines":
        def predict_power_lines():
          """ Builds (but doesn't train) the best model to predict `total_influx`.
          Returns: the model/pipeline
          """
          pass
        _check_signature(predict_power_lines, fn)
    if fn.__name__ == "generate_MEX_natural_status":
        def generate_MEX_natural_status(X3, y3, pipeline, tokenizer, length=1):
            """
            Generate natural status descriptions for the Mars Express spacecraft.

            Parameters:
            - X3: The input data
            - y3: The input labels
            - pipeline: the text processing pipeline including the Llama 2 model
            - tokenizer: the (pretrained) tokenizer
            - length: Number of rows to generate (default is 1).

            Returns:
            - List of natural status descriptions.
            """
            pass
        _check_signature(generate_MEX_natural_status, fn)
    if fn.__name__ == "create_embeddings":
        def create_embeddings(model, tokenized_sentences):
          """
          Create embeddings for a list of tokenized sentences using a pre-trained language model.

          Parameters:
          - model: Pre-trained language model capable of generating embeddings.
          - tokenized_sentences: List of tokenized sentences, where each sentence is represented as a dictionary of input tensors.

          Returns:
          - Numpy array of sentence embeddings.
          """
          pass
        _check_signature(create_embeddings, fn)
    if fn.__name__ == "compute_tsne":
        def compute_tsne(X):
          """ Applies tSNE to build a 2D representation of the data
          Returns a dataframe X with the 2D representation
          X -- The input data
          """
          pass
        _check_signature(compute_tsne, fn)
    if fn.__name__ == "plot_embeddings":
        def plot_embeddings():
          """ Uses the functions you created above to create the 2D scatter plot.
          """
          pass
        _check_signature(plot_embeddings, fn)
    if fn.__name__ == "build_final_model":
        def build_final_model():
          """ Build the best possible model (highest AUC score) for the given dataset.
          """
          pass
        _check_signature(build_final_model, fn)
