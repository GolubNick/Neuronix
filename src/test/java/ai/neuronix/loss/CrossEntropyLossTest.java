package ai.neuronix.loss;

import static ai.neuronix.helper.MotrixMock.matrix;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.within;

import ai.neuronix.math.Matrix;
import org.junit.jupiter.api.Test;

public class CrossEntropyLossTest {

  @Test
  void shouldReturnSmallLossForCorrectPrediction() {

    CrossEntropyLoss loss = new CrossEntropyLoss();

    Matrix predicted = matrix(0.1, 0.8, 0.1);
    Matrix expected = matrix(0, 1, 0);

    assertThat(loss.calculate(predicted, expected)).isLessThan(0.3);
  }

  @Test
  void shouldReturnLargeLossForWrongPrediction() {

    CrossEntropyLoss loss = new CrossEntropyLoss();

    Matrix predicted = matrix(0.8, 0.1, 0.1);
    Matrix expected = matrix(0, 1, 0);

    assertThat(loss.calculate(predicted, expected)).isGreaterThan(2.0);
  }

  @Test
  void shouldCalculateGradient() {

    CrossEntropyLoss loss = new CrossEntropyLoss();

    Matrix predicted = matrix(0.1, 0.8, 0.1);
    Matrix expected = matrix(0, 1, 0);

    Matrix gradient = loss.gradient(predicted, expected);

    assertThat(gradient.get(0, 0)).isCloseTo(0.1, within(0.000001));

    assertThat(gradient.get(1, 0)).isCloseTo(-0.2, within(0.000001));

    assertThat(gradient.get(2, 0)).isCloseTo(0.1, within(0.000001));
  }

  @Test
  void shouldHandleZeroProbability() {

    CrossEntropyLoss loss = new CrossEntropyLoss();

    Matrix predicted = matrix(0, 1, 0);
    Matrix expected = matrix(1, 0, 0);

    assertThat(loss.calculate(predicted, expected)).isFinite();
  }
}
