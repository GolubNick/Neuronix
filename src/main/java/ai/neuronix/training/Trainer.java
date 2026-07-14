package ai.neuronix.training;

import ai.neuronix.dataset.TrainingSample;
import ai.neuronix.loss.LossFunction;
import ai.neuronix.math.Matrix;
import ai.neuronix.nn.NeuralNetwork;
import java.util.List;

public class Trainer {

    private final NeuralNetwork network;
    private final LossFunction lossFunction;
    private final double learningRate;

    public Trainer(
        NeuralNetwork network,
        LossFunction lossFunction,
        double learningRate) {

        this.network = network;
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    public double train(TrainingSample sample) {

        Matrix predicted = network.forward(sample.input());

        double loss = lossFunction.calculate(
            predicted,
            sample.expected());

        Matrix gradient = lossFunction.gradient(
            predicted,
            sample.expected());

        network.backward(gradient);

        network.update(learningRate);

        return loss;
    }

    public double train(List<TrainingSample> dataset, int epochs) {

        double loss = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {

            loss = 0;

            for (TrainingSample sample : dataset) {
                loss += train(sample);
            }

            loss /= dataset.size();

        }

        return loss;
    }
}
