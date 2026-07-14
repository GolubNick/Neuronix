package ai.neuronix.helper;

import static org.assertj.core.api.Assertions.assertThat;

import ai.neuronix.math.Matrix;

public class Assertion {

    public static void assertMatrixEquals(Matrix actual, double[][] expected) {
        assertThat(actual.rows()).isEqualTo(expected.length);
        assertThat(actual.columns()).isEqualTo(expected[0].length);

        for (int row = 0; row < expected.length; row++) {
            for (int column = 0; column < expected[row].length; column++) {
                assertThat(actual.get(row, column))
                    .isEqualTo(expected[row][column]);
            }
        }
    }

}
