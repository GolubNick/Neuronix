package ai.neuronix.nn;

import ai.neuronix.math.Matrix;

public interface Layer {

    Matrix forward(Matrix input);
    Matrix backward(Matrix gradient);

}
