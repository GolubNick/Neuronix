package ai.neuronix.optimizer;

import ai.neuronix.math.MatrixOperations;

import java.util.List;

public final class SGDOptimizer implements Optimizer {

  private final double learningRate;

  public SGDOptimizer(double learningRate) {
    this.learningRate = learningRate;
  }

  @Override
  public void step(List<Parameter> parameters) {

    for (Parameter parameter : parameters) {

      parameter.value(
          MatrixOperations.subtract(
              parameter.value(),
              MatrixOperations.multiply(
                  parameter.gradient(),
                  learningRate)));
    }
  }
}