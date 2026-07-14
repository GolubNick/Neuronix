package ai.neuronix.metrics;

import ai.neuronix.math.Matrix;

public interface Metric {

  double calculate(Matrix predicted, Matrix expected);

  default void validateDimensions(Matrix predicted, Matrix expected) {

    if (predicted.rows() != expected.rows() || predicted.columns() != expected.columns()) {

      throw new IllegalArgumentException(
          "Predicted and expected matrices must have the same dimensions.");
    }
  }
}
