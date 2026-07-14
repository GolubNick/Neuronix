package ai.neuronix.nn;

import ai.neuronix.math.Matrix;
import ai.neuronix.math.MatrixOperations;

class IncrementLayer implements Layer {

    @Override
    public Matrix forward(Matrix input) {
        return MatrixOperations.map(input, value -> value + 1);
    }

    @Override
    public Matrix backward(Matrix gradient) {
        throw new UnsupportedOperationException("Backward propagation is not implemented yet.");
    }
}
