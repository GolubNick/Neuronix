package ai.neuronix.nn;

import ai.neuronix.math.Matrix;
import ai.neuronix.math.MatrixFactory;
import ai.neuronix.math.MatrixOperations;

public class DenseLayer implements TrainableLayer, Trainable {

    private Matrix weights;
    private Matrix bias;

    private Matrix weightGradient;
    private Matrix biasGradient;

    private Matrix input;

    public DenseLayer(int inputSize, int outputSize) {
        this(
            MatrixFactory.random(outputSize, inputSize),
            MatrixFactory.zeros(outputSize, 1)
        );
    }

    public DenseLayer(Matrix weights, Matrix bias) {
        this.weights = weights;
        this.bias = bias;
    }

    @Override
    public Matrix forward(Matrix input) {
        if (input.rows() != weights.columns()) {
            throw new IllegalArgumentException(
                String.format(
                    "Expected input vector with %d rows but got %d.",
                    weights.columns(),
                    input.rows()));
        }

        this.input = input;
        Matrix output = MatrixOperations.multiply(weights, input);
        return MatrixOperations.add(output, bias);
    }

    @Override
    public Matrix backward(Matrix gradient) {
        weightGradient = MatrixOperations.multiply(
            gradient,
            MatrixOperations.transpose(input));

        biasGradient = gradient;

        return MatrixOperations.multiply(
            MatrixOperations.transpose(weights),
            gradient);
    }

    @Override
    public void update(double learningRate) {

        weights = MatrixOperations.subtract(
            weights,
            MatrixOperations.multiply(weightGradient, learningRate));

        bias = MatrixOperations.subtract(
            bias,
            MatrixOperations.multiply(biasGradient, learningRate));
    }

    Matrix getWeights() {
        return weights;
    }

    Matrix getBias() {
        return bias;
    }

    Matrix getWeightGradient() {
        return weightGradient;
    }

    Matrix getBiasGradient() {
        return biasGradient;
    }
}