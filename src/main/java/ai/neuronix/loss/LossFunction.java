package ai.neuronix.loss;

import ai.neuronix.math.Matrix;

public interface LossFunction {

    double calculate(Matrix predicted, Matrix expected);

    Matrix gradient(Matrix predicted, Matrix expected);

}
