package ai.neuronix.optimizer;

import ai.neuronix.math.Matrix;

public final class Parameter {

    private Matrix value;
    private Matrix gradient;

    public Parameter(Matrix value) {
        this.value = value;
    }

    public Matrix value() {
        return value;
    }

    public void value(Matrix value) {
        this.value = value;
    }

    public Matrix gradient() {
        return gradient;
    }

    public void gradient(Matrix gradient) {
        this.gradient = gradient;
    }
}
