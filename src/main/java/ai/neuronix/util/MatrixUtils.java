package ai.neuronix.util;

import ai.neuronix.math.Matrix;

public final class MatrixUtils {

  private MatrixUtils() {}

  public static int argMax(Matrix matrix) {

    int index = 0;
    double max = matrix.get(0, 0);

    for (int row = 1; row < matrix.rows(); row++) {
      if (matrix.get(row, 0) > max) {
        max = matrix.get(row, 0);
        index = row;
      }
    }

    return index;
  }
}
