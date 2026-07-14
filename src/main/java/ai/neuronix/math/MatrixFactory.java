package ai.neuronix.math;

import java.util.concurrent.ThreadLocalRandom;

public final class MatrixFactory {

  private MatrixFactory() {}

  public static Matrix zeros(int rows, int columns) {
    return new Matrix(rows, columns);
  }

  public static Matrix ones(int rows, int columns) {

    Matrix matrix = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        matrix.set(row, column, 1.0);
      }
    }

    return matrix;
  }

  public static Matrix identity(int size) {

    Matrix matrix = new Matrix(size, size);

    for (int i = 0; i < size; i++) {
      matrix.set(i, i, 1.0);
    }

    return matrix;
  }

  public static Matrix random(int rows, int columns) {
    return random(rows, columns, -1.0, 1.0);
  }

  public static Matrix random(int rows, int columns, double min, double max) {

    Matrix matrix = new Matrix(rows, columns);

    ThreadLocalRandom random = ThreadLocalRandom.current();

    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        matrix.set(row, column, random.nextDouble(min, max));
      }
    }

    return matrix;
  }

  public static Matrix xavier(int fanIn, int fanOut) {

    double limit = Math.sqrt(6.0 / (fanIn + fanOut));

    return random(fanOut, fanIn, -limit, limit);
  }

  public static Matrix he(int fanIn, int fanOut) {

    double limit = Math.sqrt(2.0 / fanIn);

    return random(fanOut, fanIn, -limit, limit);
  }
}
