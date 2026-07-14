package ai.neuronix.nn;

import ai.neuronix.math.Matrix;

public final class TanhLayer extends ActivationLayer {

  @Override
  protected double activate(double value) {
    return Math.tanh(value);
  }

  @Override
  protected double derivative(int row, int column) {
    return 0;
  }

  @Override
  public Matrix backward(Matrix gradient) {
    return null;
  }
}
