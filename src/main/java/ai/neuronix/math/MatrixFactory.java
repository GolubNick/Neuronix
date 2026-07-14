package ai.neuronix.math;

import java.util.concurrent.ThreadLocalRandom;

public final class MatrixFactory {

    private MatrixFactory() {
    }

    public static Matrix zeros(int rows, int columns) {
        return new Matrix(rows, columns);
    }

    public static Matrix random(int rows, int columns) {

        Matrix matrix = new Matrix(rows, columns);

        ThreadLocalRandom random = ThreadLocalRandom.current();

        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                matrix.set(row, column, random.nextDouble(-1.0, 1.0));
            }
        }

        return matrix;
    }
}
