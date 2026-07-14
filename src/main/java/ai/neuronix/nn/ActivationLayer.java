package ai.neuronix.nn;

import ai.neuronix.math.Matrix;
import ai.neuronix.math.MatrixOperations;

public abstract class ActivationLayer implements Layer {
    protected Matrix input;

    @Override
    public Matrix forward(Matrix input) {
        this.input = input;
        return MatrixOperations.map(input, this::activate);
    }

    protected abstract double activate(double value);
}
