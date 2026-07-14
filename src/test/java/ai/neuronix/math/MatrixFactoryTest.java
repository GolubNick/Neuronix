package ai.neuronix.math;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

import org.junit.jupiter.api.Test;

public class MatrixFactoryTest {

  @Test
  void shouldCreateZeroMatrix() {
    Matrix matrix = MatrixFactory.zeros(3, 2);

    assertThat(matrix.rows()).isEqualTo(3);
    assertThat(matrix.columns()).isEqualTo(2);

    for (int row = 0; row < matrix.rows(); row++) {
      for (int column = 0; column < matrix.columns(); column++) {
        assertThat(matrix.get(row, column)).isZero();
      }
    }
  }

  @Test
  void shouldCreateRandomMatrixWithCorrectDimensions() {
    Matrix matrix = MatrixFactory.random(4, 5);

    assertThat(matrix.rows()).isEqualTo(4);
    assertThat(matrix.columns()).isEqualTo(5);
  }

  @Test
  void shouldCreateRandomMatrixWithValuesInRange() {
    Matrix matrix = MatrixFactory.random(20, 20);

    for (int row = 0; row < matrix.rows(); row++) {
      for (int column = 0; column < matrix.columns(); column++) {
        assertThat(matrix.get(row, column)).isGreaterThanOrEqualTo(-1.0).isLessThan(1.0);
      }
    }
  }

  @Test
  void shouldGenerateDifferentRandomMatrices() {
    Matrix first = MatrixFactory.random(10, 10);
    Matrix second = MatrixFactory.random(10, 10);

    boolean hasDifference = false;

    outer:
    for (int row = 0; row < first.rows(); row++) {
      for (int column = 0; column < first.columns(); column++) {
        if (first.get(row, column) != second.get(row, column)) {
          hasDifference = true;
          break outer;
        }
      }
    }

    assertThat(hasDifference).isTrue();
  }
}
