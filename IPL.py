import numpy as np
from scipy.optimize import curve_fit
from vertical_model_evaluator import VerticalModelEvaluator

class IPL(VerticalModelEvaluator):
    def __init__(self, config_space, surrogate_model, anchors=None, max_anchor_size=16000):
        """
        IPL-based learning curve extrapolation for early stopping
        
        :param config_space: Configuration space for hyperparameters
        :param surrogate_model: Trained surrogate model that predicts error rates
        :param anchors: Fixed schedule of training set sizes to evaluate
        :param max_anchor_size: Maximum anchor size for this dataset (target for prediction)
        """
        self.config_space = config_space
        self.surrogate_model = surrogate_model
        self.max_anchor_size = max_anchor_size
        
        # anchor selection based on dataset characteristics
        if anchors is None:
            self.anchors = self._generate_anchors(max_anchor_size)
        else:
            # Filter anchors to not exceed max_anchor_size
            self.anchors = [a for a in anchors if a <= max_anchor_size]
            
        # Track best error seen so far 
        self.best_error = float('inf')
        self.evaluation_history = []

    def _generate_anchors(self, max_anchor):
        """Generate appropriate anchor schedule based on max anchor size"""
        if max_anchor <= 256:
            # Small dataset
            return [16, 32, 64, 128, 256]
        elif max_anchor <= 2048:
            # Medium dataset
            return [16, 32, 64, 128, 256, 512, 1024, 2048]
        else:
            # Large dataset - geometric progression up to max_anchor
            anchors = []
            anchor = 16
            while anchor <= max_anchor:
                anchors.append(anchor)
                anchor = min(anchor * 2, max_anchor)
                if anchor == anchors[-1]:  # Prevent infinite loop
                    break
            return anchors

    def ipl_model(self, x, a, b, c):
        """Inverse Power Law model for error rates: error = a + b * x^(-c)"""
        return a + b * np.power(x, -c)
    
    def fit_ipl(self, anchors, errors):
        """Fit IPL model to observed learning curve data"""
        try:
            anchors_arr = np.array(anchors, dtype=np.float64)
            errors_arr = np.array(errors, dtype=np.float64)
            
            bounds = ([0.0, 0.0, 0.01], [1.0, 100.0, 10.0])  
            
            # initial guess based on the data pattern
            if len(errors_arr) >= 3:
                # Estimate parameters from the data trend
                a_guess = max(0.0, errors_arr[-1] * 0.8)  # Conservative asymptote
                b_guess = (errors_arr[0] - errors_arr[-1]) * anchors_arr[0]  # Scale based on improvement
                c_guess = 0.5  # Moderate decay
            else:
                a_guess, b_guess, c_guess = errors_arr[-1], 1.0, 0.5
                
            p0 = [a_guess, b_guess, c_guess]
            
            params, _ = curve_fit(self.ipl_model, anchors_arr, errors_arr, 
                                p0=p0, bounds=bounds, maxfev=5000)
            return params
            
        except (RuntimeError, ValueError) as e:
            print(f"IPL fitting failed: {e}. Using conservative estimates.")
            # Return parameters that at least follow the trend
            return [errors[-1], (errors[0] - errors[-1]) * 10, 0.5]
    
    def evaluate(self, configuration):
        """
        Evaluate a configuration using IPL extrapolation with early stopping
        Uses the class's max_anchor_size as the target size
        """
        observed_anchors = []
        observed_errors = []
        stopped_at_anchor = None
        
        print(f"Evaluating configuration")
        print(f"Anchor schedule: {self.anchors}")
        print(f"Target size: {self.max_anchor_size}")
        
        for i, anchor in enumerate(self.anchors):
            try:
                # Get error prediction from surrogate model
                error = self.surrogate_model.predict(configuration, anchor=anchor)
                observed_anchors.append(anchor)
                observed_errors.append(error)
                
                print(f"  Anchor {anchor}: error = {error:.4f}")
                
                # Check if we have enough points to start extrapolation
                if len(observed_errors) >= 3:
                    # Fit IPL to current observations
                    a, b, c = self.fit_ipl(observed_anchors, observed_errors)
                    
                    # Predict error at max anchor size for this dataset
                    predicted_error = self.ipl_model(self.max_anchor_size, a, b, c)
                    
                    print(f"    IPL prediction at {self.max_anchor_size}: {predicted_error:.4f} (best: {self.best_error:.4f})")
                    
                    # Early stopping decision
                    margin = 0.01  # 1% margin to avoid being too aggressive
                    if predicted_error > self.best_error + margin:
                        print(f"    → DISCARDING: predicted too high")
                        stopped_at_anchor = anchor
                        
                        self.evaluation_history.append({
                            'configuration': configuration,
                            'stopped_at_anchor': stopped_at_anchor,
                            'reason': 'IPL_prediction_worse',
                            'final_prediction': None,
                            'observed_anchors': observed_anchors.copy(),  
                            'observed_errors': observed_errors.copy()     
                        })
                        return None
                        
            except Exception as e:
                print(f"    Error at anchor {anchor}: {e}")
                continue
        
        # If we complete all anchors without being stopped
        if len(observed_errors) >= 3:
            a, b, c = self.fit_ipl(observed_anchors, observed_errors)
            final_prediction = self.ipl_model(self.max_anchor_size, a, b, c)
        else:
            # Not enough points for reliable extrapolation
            final_prediction = observed_errors[-1]
        
        print(f"  Final prediction: {final_prediction:.4f}")
        
        # Update best error if improved
        if final_prediction < self.best_error:
            old_best = self.best_error
            self.best_error = final_prediction
            print(f"  → NEW BEST ERROR: {final_prediction:.4f} (improved from {old_best:.4f})")
        
        self.evaluation_history.append({
            'configuration': configuration,
            'stopped_at_anchor': None,
            'reason': 'completed_all_anchors',
            'final_prediction': final_prediction,
            'observed_anchors': observed_anchors,
            'observed_errors': observed_errors
        })
        
        return final_prediction
    
    def get_anchor_schedule(self):
        """Return the anchor schedule being used"""
        return self.anchors
    
    def reset(self):
        """Reset the evaluator state for new experiments"""
        self.best_error = float('inf')
        self.evaluation_history = []
    
    def update_best_performance(self, error):
        """
        Update the best performance seen so far
        Useful when you have external information about good configurations
        """
        if error < self.best_error:
            self.best_error = error
    
    def get_evaluation_stats(self):
        """Get statistics about evaluations performed"""
        total_evaluations = len(self.evaluation_history)
        stopped_early = len([e for e in self.evaluation_history if e['stopped_at_anchor'] is not None])
        completed = total_evaluations - stopped_early
        
        return {
            'total_evaluations': total_evaluations,
            'stopped_early': stopped_early,
            'completed': completed,
            'early_stopping_rate': stopped_early / total_evaluations if total_evaluations > 0 else 0,
            'best_error': self.best_error
        }