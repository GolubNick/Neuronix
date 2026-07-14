package ai.neuronix.math;

import java.util.Arrays;

public final class Matrix {

  private final int rows;
  private final int columns;
  private final double[][] values;

  public Matrix(int rows, int columns) {
    if (rows <= 0 || columns <= 0) {
      throw new IllegalArgumentException("Matrix dimensions must be positive.");
    }

    this.rows = rows;
    this.columns = columns;
    this.values = new double[rows][columns];
  }

  public int rows() {
    return rows;
  }

  public int columns() {
    return columns;
  }

  public double get(int row, int column) {
    return values[row][column];
  }

  public void set(int row, int column, double value) {

    if (Double.isNaN(value)) {
      throw new IllegalStateException("NaN written to matrix");
    }

    if (Double.isInfinite(value)) {
      throw new IllegalStateException("Infinity written to matrix");
    }

    values[row][column] = value;
  }

  @Override
  public String toString() {
    return Arrays.deepToString(values);
  }
}
