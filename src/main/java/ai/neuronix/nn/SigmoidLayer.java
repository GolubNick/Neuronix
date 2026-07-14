package ai.neuronix.nn;

import ai.neuronix.math.Matrix;

public final class SigmoidLayer extends ActivationLayer {

    @Override
    protected double activate(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    @Override
    public Matrix backward(Matrix gradient) {
        return null;
    }
}
