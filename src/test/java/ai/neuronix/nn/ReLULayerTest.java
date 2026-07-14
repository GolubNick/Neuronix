package ai.neuronix.nn;

import static ai.neuronix.helper.Assertion.assertMatrixEquals;
import static ai.neuronix.helper.MotrixMock.matrix;

import ai.neuronix.math.Matrix;
import org.junit.jupiter.api.Test;

public class ReLULayerTest {

    @Test
    void shouldApplyRelu() {

        ReLULayer relu = new ReLULayer();

        Matrix input = matrix(new double[][]{
            {-2},
            {-1},
            {0},
            {5}
        });

        Matrix output = relu.forward(input);

        assertMatrixEquals(output, new double[][]{
            {0},
            {0},
            {0},
            {5}
        });
    }

}
