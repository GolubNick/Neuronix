package ai.neuronix.nn;

import ai.neuronix.math.Matrix;

public final class ReLULayer extends ActivationLayer {

    @Override
    protected double activate(double value) {
        return Math.max(0.0, value);
    }

    @Override
    public Matrix backward(Matrix gradient) {
        Matrix result = new Matrix(input.rows(), input.columns());

        for (int row = 0; row < input.rows(); row++) {
            for (int column = 0; column < input.columns(); column++) {

                double derivative = input.get(row, column) > 0 ? 1.0 : 0.0;

                result.set(
                    row,
                    column,
                    gradient.get(row, column) * derivative);
            }
        }

        return result;
    }
}
