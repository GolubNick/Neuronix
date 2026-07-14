package ai.neuronix.math;

import ai.neuronix.nn.DenseLayer;
import org.junit.jupiter.api.Test;

import static ai.neuronix.helper.Assertion.assertMatrixEquals;
import static ai.neuronix.helper.MotrixMock.matrix;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class MatrixOperationsTest {

    @Test
    void shouldAddMatrices() {
        Matrix left = matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix right = matrix(new double[][]{
            {5, 6},
            {7, 8}
        });

        Matrix result = MatrixOperations.add(left, right);

        assertMatrixEquals(result, new double[][]{
            {6, 8},
            {10, 12}
        });
    }

    @Test
    void shouldAddMatricesWithNegativeValues() {
        Matrix left = matrix(new double[][]{
            {-1, 2, -3}
        });

        Matrix right = matrix(new double[][]{
            {4, -5, 6}
        });

        Matrix result = MatrixOperations.add(left, right);

        assertMatrixEquals(result, new double[][]{
            {3, -3, 3}
        });
    }

    @Test
    void shouldThrowExceptionWhenMatrixDimensionsDoNotMatch() {
        Matrix left = new Matrix(2, 2);
        Matrix right = new Matrix(3, 2);

        assertThatThrownBy(() -> MatrixOperations.add(left, right))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Matrices must have the same dimensions.");
    }

    @Test
    void shouldMultiplyMatrices() {

        Matrix left = matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix right = matrix(new double[][]{
            {5, 6},
            {7, 8}
        });

        Matrix result = MatrixOperations.multiply(left, right);

        assertMatrixEquals(result, new double[][]{
            {19, 22},
            {43, 50}
        });
    }

    @Test
    void shouldMultiplyRectangularMatrices() {

        Matrix left = matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });

        Matrix right = matrix(new double[][]{
            {7, 8},
            {9, 10},
            {11, 12}
        });

        Matrix result = MatrixOperations.multiply(left, right);

        assertMatrixEquals(result, new double[][]{
            {58, 64},
            {139, 154}
        });
    }

    @Test
    void shouldThrowExceptionWhenMatricesCannotBeMultiplied() {

        Matrix left = new Matrix(2, 3);
        Matrix right = new Matrix(2, 2);

        assertThatThrownBy(() -> MatrixOperations.multiply(left, right))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("The number of columns in the left matrix must be equal to the number of rows in the right matrix.");
    }

    @Test
    void shouldTransposeMatrix() {

        Matrix matrix = matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });

        Matrix result = MatrixOperations.transpose(matrix);

        assertMatrixEquals(result, new double[][]{
            {1, 4},
            {2, 5},
            {3, 6}
        });
    }

    @Test
    void shouldSubtractMatrices() {

        Matrix left = matrix(new double[][]{
            {5, 6},
            {7, 8}
        });

        Matrix right = matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix result = MatrixOperations.subtract(left, right);

        assertMatrixEquals(result, new double[][]{
            {4, 4},
            {4, 4}
        });
    }

    @Test
    void shouldMapMatrix() {

        Matrix matrix = matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix result = MatrixOperations.map(matrix, value -> value * 10);

        assertMatrixEquals(result, new double[][]{
            {10, 20},
            {30, 40}
        });
    }

    @Test
    void shouldPerformForwardPass() {

        Matrix weights = matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix bias = matrix(new double[][]{
            {1},
            {2}
        });

        DenseLayer layer = new DenseLayer(weights, bias);

        Matrix input = matrix(new double[][]{
            {5},
            {6}
        });

        Matrix output = layer.forward(input);

        assertMatrixEquals(output, new double[][]{
            {18},
            {41}
        });
    }

    @Test
    void shouldMultiplyMatrixByScalar() {

        Matrix matrix = matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        Matrix result = MatrixOperations.multiply(matrix, 0.5);

        assertMatrixEquals(result, new double[][]{
            {0.5, 1},
            {1.5, 2}
        });
    }

}
