package ai.neuronix.nn;

import static ai.neuronix.helper.MotrixMock.matrix;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;
import static org.assertj.core.api.AssertionsForClassTypes.within;

import ai.neuronix.math.Matrix;
import org.junit.jupiter.api.Test;

public class SoftmaxLayerTest {

  @Test
  void shouldReturnProbabilitiesThatSumToOne() {
    SoftmaxLayer layer = new SoftmaxLayer();
    Matrix output = layer.forward(matrix(2, 4, 1));
    double sum = 0;
    for (int row = 0; row < output.rows(); row++) {
      sum += output.get(row, 0);
    }
    assertThat(sum).isCloseTo(1.0, within(0.000001));
  }

  @Test
  void shouldReturnValuesBetweenZeroAndOne() {
    SoftmaxLayer layer = new SoftmaxLayer();
    Matrix output = layer.forward(matrix(2, 4, 1));
    for (int row = 0; row < output.rows(); row++) {
      assertThat(output.get(row, 0)).isBetween(0.0, 1.0);
    }
  }

  @Test
  void shouldAssignHighestProbabilityToLargestInput() {
    SoftmaxLayer layer = new SoftmaxLayer();
    Matrix output = layer.forward(matrix(2, 4, 1));
    assertThat(output.get(1, 0)).isGreaterThan(output.get(0, 0));
    assertThat(output.get(1, 0)).isGreaterThan(output.get(2, 0));
  }

  @Test
  void shouldBeNumericallyStable() {
    SoftmaxLayer layer = new SoftmaxLayer();
    Matrix output = layer.forward(matrix(1000, 1001, 999));
    double sum = 0;
    for (int row = 0; row < output.rows(); row++) {
      sum += output.get(row, 0);
    }
    assertThat(sum).isCloseTo(1.0, within(0.000001));
    for (int row = 0; row < output.rows(); row++) {
      assertThat(output.get(row, 0)).isFinite();
    }
  }

  @Test
  void shouldRejectMatrixInput() {
    SoftmaxLayer layer = new SoftmaxLayer();
    assertThatThrownBy(
            () ->
                layer.forward(
                    matrix(
                        new double[][] {
                          {1, 2},
                          {3, 4}
                        })))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessage("Softmax expects a column vector.");
  }
}
