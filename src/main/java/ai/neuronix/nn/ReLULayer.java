package ai.neuronix.nn;

public final class ReLULayer extends ActivationLayer {

  @Override
  protected double activate(double value) {
    return Math.max(0.0, value);
  }

  @Override
  protected double derivative(int row, int column) {
    return getInput().get(row, column) > 0 ? 1.0 : 0.0;
  }
}
