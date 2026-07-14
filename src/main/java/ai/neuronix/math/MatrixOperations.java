package ai.neuronix.math;

import java.util.function.DoubleUnaryOperator;

public final class MatrixOperations {

    private MatrixOperations() {
    }

    public static Matrix add(Matrix left, Matrix right) {

        validateSameDimensions(left, right);

        Matrix result = new Matrix(
            left.rows(),
            left.columns());

        for (int row = 0; row < left.rows(); row++) {
            for (int column = 0; column < left.columns(); column++) {

                result.set(
                    row,
                    column,
                    left.get(row, column)
                        + right.get(row, column));
            }
        }

        return result;
    }


    public static Matrix multiply(Matrix left, Matrix right) {

        validateMultiplicationDimensions(left, right);

        Matrix result = new Matrix(left.rows(), right.columns());

        for (int row = 0; row < left.rows(); row++) {

            for (int column = 0; column < right.columns(); column++) {

                double sum = 0;

                for (int i = 0; i < left.columns(); i++) {
                    sum += left.get(row, i) * right.get(i, column);
                }

                result.set(row, column, sum);
            }
        }

        return result;
    }

    public static Matrix multiply(Matrix matrix, double scalar) {

        Matrix result = new Matrix(matrix.rows(), matrix.columns());

        for (int row = 0; row < matrix.rows(); row++) {
            for (int column = 0; column < matrix.columns(); column++) {

                result.set(
                    row,
                    column,
                    matrix.get(row, column) * scalar);
            }
        }

        return result;
    }

    public static Matrix transpose(Matrix matrix) {

        Matrix result = new Matrix(matrix.columns(), matrix.rows());

        for (int row = 0; row < matrix.rows(); row++) {
            for (int column = 0; column < matrix.columns(); column++) {
                result.set(column, row, matrix.get(row, column));
            }
        }

        return result;
    }

    public static Matrix subtract(Matrix left, Matrix right) {
        validateSameDimensions(left, right);

        Matrix result = new Matrix(left.rows(), left.columns());

        for (int row = 0; row < left.rows(); row++) {
            for (int column = 0; column < left.columns(); column++) {
                result.set(
                    row,
                    column,
                    left.get(row, column) - right.get(row, column));
            }
        }

        return result;
    }

    public static Matrix map(Matrix matrix, DoubleUnaryOperator mapper) {

        Matrix result = new Matrix(matrix.rows(), matrix.columns());

        for (int row = 0; row < matrix.rows(); row++) {
            for (int column = 0; column < matrix.columns(); column++) {
                result.set(
                    row,
                    column,
                    mapper.applyAsDouble(matrix.get(row, column)));
            }
        }

        return result;
    }

    private static void validateMultiplicationDimensions(Matrix left, Matrix right) {
        if (left.columns() != right.rows()) {
            throw new IllegalArgumentException(
                String.format(
                    "Cannot multiply matrices with dimensions %dx%d and %dx%d.",
                    left.rows(),
                    left.columns(),
                    right.rows(),
                    right.columns()));
        }
    }

    private static void validateSameDimensions(Matrix left, Matrix right) {

        if (left.rows() != right.rows()
            || left.columns() != right.columns()) {

            throw new IllegalArgumentException(
                String.format(
                    "Matrices must have the same dimensions: %dx%d and %dx%d.",
                    left.rows(),
                    left.columns(),
                    right.rows(),
                    right.columns()));
        }
    }

}
