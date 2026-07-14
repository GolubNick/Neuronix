package ai.neuronix.nn;

import ai.neuronix.math.Matrix;

public class SoftmaxLayer implements Layer {
  private Matrix output;

  @Override
  public Matrix forward(Matrix input) {
    if (input.columns() != 1) {
      throw new IllegalArgumentException("Softmax expects a column vector.");
    }

    double max = Double.NEGATIVE_INFINITY;

    for (int row = 0; row < input.rows(); row++) {
      max = Math.max(max, input.get(row, 0));
    }

    Matrix output = new Matrix(input.rows(), input.columns());

    double sum = 0.0;

    for (int row = 0; row < input.rows(); row++) {

      double value = Math.exp(input.get(row, 0) - max);

      output.set(row, 0, value);

      sum += value;
    }

    for (int row = 0; row < output.rows(); row++) {
      output.set(row, 0, output.get(row, 0) / sum);
    }

    this.output = output;

    return output;
  }

  @Override
  public Matrix backward(Matrix gradient) {
    return gradient;
  }
}
