package ai.neuronix.loss;

import ai.neuronix.math.Matrix;
import ai.neuronix.math.MatrixOperations;

public final class CrossEntropyLoss implements LossFunction {

  private static final double EPSILON = 1e-15;

  @Override
  public double calculate(Matrix predicted, Matrix expected) {

    validateDimensions(predicted, expected);

    double loss = 0.0;

    for (int row = 0; row < predicted.rows(); row++) {

      double probability = Math.max(predicted.get(row, 0), EPSILON);

      loss -= expected.get(row, 0) * Math.log(probability);
    }

    return loss;
  }

  @Override
  public Matrix gradient(Matrix predicted, Matrix expected) {

    validateDimensions(predicted, expected);

    return MatrixOperations.subtract(predicted, expected);
  }

  private void validateDimensions(Matrix predicted, Matrix expected) {

    if (predicted.rows() != expected.rows() || predicted.columns() != expected.columns()) {

      throw new IllegalArgumentException(
          "Predicted and expected matrices must have the same dimensions.");
    }
  }
}
