package ai.neuronix.loss;

import static ai.neuronix.helper.MotrixMock.matrix;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

import ai.neuronix.math.Matrix;
import org.junit.jupiter.api.Test;

public class MeanSquaredErrorLossTest {

  @Test
  void shouldCalculateMeanSquaredError() {

    MeanSquaredErrorLoss loss = new MeanSquaredErrorLoss();

    Matrix predicted = matrix(new double[][] {{2}});

    Matrix expected = matrix(new double[][] {{3}});

    assertThat(loss.calculate(predicted, expected)).isEqualTo(1.0);
  }

  @Test
  void shouldCalculateMeanSquaredErrorForVector() {

    MeanSquaredErrorLoss loss = new MeanSquaredErrorLoss();

    Matrix predicted = matrix(new double[][] {{1}, {2}});

    Matrix expected = matrix(new double[][] {{2}, {4}});

    assertThat(loss.calculate(predicted, expected)).isEqualTo(2.5);
  }
}
