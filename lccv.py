import logging
import numpy as np
import typing
from sklearn.pipeline import Pipeline

from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):

    def __init__(self, surrogate_model: Pipeline, minimal_anchor: int, final_anchor: int) -> None:
        """
        Initialises the LCCV evaluator.
        
        This method calls the parent constructor and then creates the list
        of anchors (self.anchors) that will be evaluated.
        """
        # Call the parent __init__
        super().__init__(surrogate_model, minimal_anchor, final_anchor)
        
        # Create the list of anchors, e.g., [256, 512, 1024, ..., final_anchor]
        anchors = []
        current_anchor = minimal_anchor
        while current_anchor < final_anchor:
            anchors.append(current_anchor)
            current_anchor *= 2 
        # Ensure the final anchor is always the last item
        if final_anchor not in anchors:
            anchors.append(final_anchor)
        self.anchors = anchors
        logging.info(f"LCCV initialized with anchors: {self.anchors}")

    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int, previous_performance: float, 
        current_anchor: int, current_performance: float, target_anchor: int
    ) -> float:
        """
        Does the optimistic performance. Since we are working with a simplified
        surrogate model, we can not measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param previous_anchor: See name
        :param previous_performance: Performance at previous anchor
        :param current_anchor: See name
        :param current_performance: Performance at current anchor
        :param target_anchor: the anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        C_t = current_performance
        s_t = current_anchor
        C_t_minus_1 = previous_performance
        s_t_minus_1 = previous_anchor
        s_T = target_anchor
        if (s_t_minus_1 - s_t) == 0:
            logging.warning("Previous and current anchors are the same. Cannot extrapolate.")
            return C_t
        slope = (C_t_minus_1 - C_t) / (s_t_minus_1 - s_t)
        extrapolated_performance = C_t + (s_T - s_t) * slope
        return extrapolated_performance
    
    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        """
        Does a staged evaluation of the model, on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. In case the optimistic extrapolation can not improve
        over the best so far, it stops the evaluation.
        In case the best so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor)

        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """
        results_list: typing.List[typing.Tuple[int, float]] = []
        # Case 1: No previous best. Evaluate on the full dataset.
        if best_so_far is None:
            # self.final_anchor is inherited from VerticalModelEvaluator
            performance = self.surrogate_model.predict(configuration, self.final_anchor)
            results_list.append((self.final_anchor, performance))
            return results_list

        # We have a 'best_so_far' and maybe need to prune.
        # self.anchors is defined in VerticalModelEvaluator (just by doubling sizes)
        for i, current_anchor in enumerate(self.anchors):
            current_perf = self.surrogate_model.predict(configuration, current_anchor)
            results_list.append((current_anchor, current_perf))

            # We need at least two points to extrapolate (i > 0)
            if i > 0:
                previous_anchor = self.anchors[i-1]
                previous_perf = results_list[i-1][1] # Get performance from the list

                # Predict performance at the *final* anchor
                extrapolated_perf = self.optimistic_extrapolation(
                    previous_anchor=previous_anchor,
                    previous_performance=previous_perf,
                    current_anchor=current_anchor,
                    current_performance=current_perf,
                    target_anchor=self.final_anchor
                )

                # Pruning Check:
                # We assume lower is better (e.g., error rate).
                # If the "optimistic" prediction is *still worse* than the
                # best we've already found, stop.
                if extrapolated_perf >= best_so_far:
                    logging.info(f"Pruning config. Extrapolated: {extrapolated_perf:.4f} >= Best: {best_so_far:.4f}")
                    # Stop early and return the results we have so far
                    return results_list
    
        return results_list
