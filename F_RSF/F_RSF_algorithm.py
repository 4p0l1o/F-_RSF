from collections import OrderedDict

from plato.algorithms import base
from plato.trainers.base import Trainer
import numpy as np


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""

    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
        self.weights = None


    def compute_weight_deltas(self, baseline_weights, weights_received):
        """Compute the deltas between baseline weights and weights received."""
        # Calculate updates from the received weights
        deltas = []
        for weight in weights_received:
            delta = OrderedDict()
            if hasattr(weight, 'items'):
                for name, current_weight in weight.items():
                    baseline = baseline_weights[name]

                    # Calculate update
                    _delta = current_weight - baseline
                    delta[name] = _delta
                deltas.append(delta)

        return deltas

    def update_weights(self, weights):
        """Updates the existing model weights from the provided deltas."""
        for weight in weights:
            if self.weights is not None:
                print(self.weights["ci"])
                ibs_trees = self.weights["ibs"]
                ibs_trees.extend(weight[0])
                ci_trees = self.weights["ci"]
                ci_trees.extend(weight[1])


                feature_names = self.weights["feature_names"]
                #np.concatenate((feature_names, weight[1]))
                event_times = self.weights["event_times"]
                event_times = np.concatenate((event_times, weight[3]))
                event_times = np.sort(event_times)
                output = self.weights["output"]
                output = output + weight[4]
                self.weights = {
                    "ibs": ibs_trees,
                    "ci": ci_trees,
                    "feature_names": feature_names,
                    "event_times":event_times,
                    "output": output 
                }
            else:
                print(weight)
                print(type(weight))
                print(weight[0])
                self.weights = {
                    "ibs": weight[0],
                    "ci": weight[1],
                    "feature_names": weight[2],
                    "event_times": weight[3],
                    "output": weight[4]
                }
        return self.weights

    def extract_weights(self, model=None):
        """Extracts weights from the model."""
        return self.weights

    def load_weights(self, weights):
        """Loads the model weights passed in as a parameter."""
        #self.model.load_state_dict(weights, strict=True)
        #print(weights)