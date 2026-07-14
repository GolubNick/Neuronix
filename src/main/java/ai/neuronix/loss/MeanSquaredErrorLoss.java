package ai.neuronix.loss;

import ai.neuronix.math.Matrix;

public class MeanSquaredErrorLoss implements LossFunction {

    @Override
    public double calculate(Matrix predicted, Matrix expected) {

        validateDimensions(predicted, expected);

        double sum = 0;

        for (int row = 0; row < predicted.rows(); row++) {

            double difference = predicted.get(row, 0)
                - expected.get(row, 0);

            sum += difference * difference;
        }

        return sum / predicted.rows();
    }

    @Override
    public Matrix gradient(Matrix predicted, Matrix expected) {

        validateDimensions(predicted, expected);

        Matrix gradient = new Matrix(predicted.rows(), 1);

        for (int row = 0; row < predicted.rows(); row++) {

            double value =
                2.0 *
                    (predicted.get(row, 0) - expected.get(row, 0))
                    / predicted.rows();

            gradient.set(row, 0, value);
        }

        return gradient;
    }

    private void validateDimensions(Matrix predicted, Matrix expected) {

        if (predicted.rows() != expected.rows()
            || predicted.columns() != expected.columns()) {

            throw new IllegalArgumentException(
                "Predicted and expected matrices must have the same dimensions.");
        }
    }
}
