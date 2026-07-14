package ai.neuronix.helper;

import ai.neuronix.math.Matrix;

public class MotrixMock {

    public static Matrix matrix(double[][] values) {
        Matrix matrix = new Matrix(values.length, values[0].length);

        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                matrix.set(row, column, values[row][column]);
            }
        }

        return matrix;
    }

}
