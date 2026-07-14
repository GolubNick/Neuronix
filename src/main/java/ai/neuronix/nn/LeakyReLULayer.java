package ai.neuronix.nn;

import ai.neuronix.math.Matrix;

public final class LeakyReLULayer extends ActivationLayer {

    @Override
    protected double activate(double value) {
        return value > 0 ? value : value * 0.01;
    }

    @Override
    public Matrix backward(Matrix gradient) {
        return null;
    }
}