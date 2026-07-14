package ai.neuronix.metrics;

import ai.neuronix.math.Matrix;
import ai.neuronix.util.MatrixUtils;

public final class Accuracy implements Metric {

  @Override
  public double calculate(Matrix predicted, Matrix expected) {

    validateDimensions(predicted, expected);

    return MatrixUtils.argMax(predicted) == MatrixUtils.argMax(expected) ? 1.0 : 0.0;
  }
}
