package ai.neuronix.integration;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.within;

import ai.neuronix.dataset.TrainingSample;
import ai.neuronix.loss.LossFunction;
import ai.neuronix.loss.MeanSquaredErrorLoss;
import ai.neuronix.math.Matrix;
import ai.neuronix.nn.DenseLayer;
import ai.neuronix.nn.NeuralNetwork;
import ai.neuronix.nn.ReLULayer;
import ai.neuronix.training.Trainer;
import java.util.List;
import org.junit.jupiter.api.Test;

class NeuralNetworkTrainingTest {

    @Test
    void shouldLearnSimpleAddition() {

        NeuralNetwork network = new NeuralNetwork()
            .add(new DenseLayer(2, 1));

        LossFunction loss = new MeanSquaredErrorLoss();

        Matrix input = matrix(new double[][]{
            {1},
            {1}
        });

        Matrix expected = matrix(new double[][]{
            {2}
        });

        Matrix predicted = network.forward(input);

        double initialLoss = loss.calculate(predicted, expected);

        for (int epoch = 0; epoch < 1000; epoch++) {

            predicted = network.forward(input);

            Matrix gradient = loss.gradient(predicted, expected);

            network.backward(gradient);

            network.update(0.01);
        }

        predicted = network.forward(input);

        double finalLoss = loss.calculate(predicted, expected);

        assertThat(finalLoss).isLessThan(initialLoss);

        assertThat(predicted.get(0, 0))
            .isCloseTo(2.0, within(0.1));
    }

    @Test
    void shouldReduceTrainingLoss() {

        NeuralNetwork network = new NeuralNetwork()
            .add(new DenseLayer(2, 4))
            .add(new ReLULayer())
            .add(new DenseLayer(4, 1));

        Trainer trainer = new Trainer(
            network,
            new MeanSquaredErrorLoss(),
            0.01);

        List<TrainingSample> dataset = List.of(
            sample(1, 1, 2),
            sample(2, 3, 5),
            sample(3, 4, 7),
            sample(5, 7, 12),
            sample(8, 2, 10),
            sample(4, 6, 10)
        );

        double before = averageLoss(network, dataset);

        trainer.train(dataset, 5000);

        double after = averageLoss(network, dataset);

        assertThat(after).isLessThan(before);
    }

    private static TrainingSample sample(double x1,
        double x2,
        double y) {

        return new TrainingSample(
            matrix(x1, x2),
            matrix(y));
    }

    private static Matrix matrix(double... values) {

        Matrix matrix = new Matrix(values.length, 1);

        for (int i = 0; i < values.length; i++) {
            matrix.set(i, 0, values[i]);
        }

        return matrix;
    }

    private static Matrix matrix(double[][] values) {
        Matrix matrix = new Matrix(values.length, values[0].length);

        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                matrix.set(row, column, values[row][column]);
            }
        }

        return matrix;
    }

    private double averageLoss(NeuralNetwork network,
        List<TrainingSample> dataset) {

        MeanSquaredErrorLoss loss = new MeanSquaredErrorLoss();

        double sum = 0;

        for (TrainingSample sample : dataset) {
            sum += loss.calculate(
                network.forward(sample.input()),
                sample.expected());
        }

        return sum / dataset.size();
    }
}
