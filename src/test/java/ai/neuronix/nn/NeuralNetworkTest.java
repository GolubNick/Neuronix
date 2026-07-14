package ai.neuronix.nn;

import static ai.neuronix.helper.Assertion.assertMatrixEquals;
import static ai.neuronix.helper.MotrixMock.matrix;

import ai.neuronix.math.Matrix;
import org.junit.jupiter.api.Test;

public class NeuralNetworkTest {

    @Test
    void shouldExecuteLayersInOrder() {

        NeuralNetwork network = new NeuralNetwork();

        network.add(new IncrementLayer())
            .add(new IncrementLayer())
            .add(new IncrementLayer());

        Matrix input = matrix(new double[][]{
            {1},
            {2}
        });

        Matrix output = network.forward(input);

        assertMatrixEquals(output, new double[][]{
            {4},
            {5}
        });
    }

}
