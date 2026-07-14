package ai.neuronix.nn;

import ai.neuronix.math.Matrix;
import ai.neuronix.math.MatrixFactory;
import ai.neuronix.math.MatrixOperations;
import ai.neuronix.optimizer.Parameter;
import ai.neuronix.optimizer.Parameterized;
import java.util.List;

public class DenseLayer implements TrainableLayer, Trainable, Parameterized {

  private final Parameter weights;
  private final Parameter bias;

  private Matrix input;

  public DenseLayer(int inputSize, int outputSize) {
    this(
        MatrixFactory.random(outputSize, inputSize),
        MatrixFactory.zeros(outputSize, 1));
  }

  public DenseLayer(Matrix weights, Matrix bias) {
    this.weights = new Parameter(weights);
    this.bias = new Parameter(bias);
  }

  @Override
  public Matrix forward(Matrix input) {

    Matrix weightMatrix = weights.value();

    if (input.rows() != weightMatrix.columns()) {
      throw new IllegalArgumentException(
          String.format(
              "Expected input vector with %d rows but got %d.",
              weightMatrix.columns(),
              input.rows()));
    }

    this.input = input;

    Matrix output = MatrixOperations.multiply(weightMatrix, input);

    return MatrixOperations.add(output, bias.value());
  }

  @Override
  public Matrix backward(Matrix gradient) {

    Matrix weightGradient =
        MatrixOperations.multiply(gradient, MatrixOperations.transpose(input));

    weights.gradient(weightGradient);
    bias.gradient(gradient);

    return MatrixOperations.multiply(
        MatrixOperations.transpose(weights.value()),
        gradient);
  }

  @Override
  public void update(double learningRate) {

    weights.value(
        MatrixOperations.subtract(
            weights.value(),
            MatrixOperations.multiply(
                weights.gradient(),
                learningRate)));

    bias.value(
        MatrixOperations.subtract(
            bias.value(),
            MatrixOperations.multiply(
                bias.gradient(),
                learningRate)));
  }

  @Override
  public List<Parameter> parameters() {
    return List.of(weights, bias);
  }

  Matrix getWeights() {
    return weights.value();
  }

  Matrix getBias() {
    return bias.value();
  }

  Matrix getWeightGradient() {
    return weights.gradient();
  }

  Matrix getBiasGradient() {
    return bias.gradient();
  }
}