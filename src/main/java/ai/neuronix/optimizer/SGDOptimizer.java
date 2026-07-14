package ai.neuronix.optimizer;

import ai.neuronix.math.MatrixOperations;
import ai.neuronix.nn.DenseLayer;

public class SGDOptimizer {

    private final double learningRate;

    public SGDOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

//    public void step(DenseLayer layer) {
//
//        layer.setWeights(
//            MatrixOperations.subtract(
//                layer.getWeights(),
//                MatrixOperations.multiply(
//                    layer.getWeightGradient(),
//                    learningRate)));
//
//        layer.setBias(
//            MatrixOperations.subtract(
//                layer.getBias(),
//                MatrixOperations.multiply(
//                    layer.getBiasGradient(),
//                    learningRate)));
//    }
}
