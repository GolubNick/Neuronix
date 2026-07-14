package ai.neuronix.nn;

public final class SigmoidLayer extends ActivationLayer {

  @Override
  protected double activate(double value) {
    return 1.0 / (1.0 + Math.exp(-value));
  }

  @Override
  protected double derivative(int row, int column) {
    double output = getOutput().get(row, column);
    return output * (1.0 - output);
  }
}
