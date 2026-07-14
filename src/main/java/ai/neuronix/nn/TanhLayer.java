package ai.neuronix.nn;

import ai.neuronix.math.Matrix;

public final class TanhLayer extends ActivationLayer {

    @Override
    protected double activate(double value) {
        return Math.tanh(value);
    }

    @Override
    public Matrix backward(Matrix gradient) {
        return null;
    }
}
