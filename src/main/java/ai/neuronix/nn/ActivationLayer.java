package ai.neuronix.nn;

import ai.neuronix.math.Matrix;
import ai.neuronix.math.MatrixOperations;

public abstract class ActivationLayer implements Layer {
  protected Matrix input;
  private Matrix output;

  @Override
  public Matrix forward(Matrix input) {
    this.input = input;
    output = MatrixOperations.map(input, this::activate);
    return output;
  }

  @Override
  public Matrix backward(Matrix gradient) {

    Matrix result = new Matrix(gradient.rows(), gradient.columns());

    for (int row = 0; row < gradient.rows(); row++) {
      for (int column = 0; column < gradient.columns(); column++) {

        result.set(row, column, gradient.get(row, column) * derivative(row, column));
      }
    }

    return result;
  }

  protected Matrix getInput() {
    return input;
  }

  protected Matrix getOutput() {
    return output;
  }

  protected abstract double activate(double value);

  protected abstract double derivative(int row, int column);
}
