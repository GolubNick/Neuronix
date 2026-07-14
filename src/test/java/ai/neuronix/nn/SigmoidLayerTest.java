package ai.neuronix.nn;

import static ai.neuronix.helper.MotrixMock.matrix;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.within;

import ai.neuronix.math.Matrix;
import org.junit.jupiter.api.Test;

public class SigmoidLayerTest {

  @Test
  void shouldApplySigmoid() {
    SigmoidLayer layer = new SigmoidLayer();
    Matrix output = layer.forward(matrix(-1, 0, 1));
    assertThat(output.get(0, 0)).isCloseTo(0.2689, within(0.0001));
    assertThat(output.get(1, 0)).isCloseTo(0.5, within(0.0001));
    assertThat(output.get(2, 0)).isCloseTo(0.7311, within(0.0001));
  }

  @Test
  void shouldCalculateSigmoidGradient() {
    SigmoidLayer layer = new SigmoidLayer();
    layer.forward(matrix(0));
    Matrix gradient = layer.backward(matrix(2));
    assertThat(gradient.get(0, 0)).isCloseTo(0.5, within(0.0001));
  }
}
