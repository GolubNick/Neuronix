package ai.neuronix.integration;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

import ai.neuronix.dataset.TrainingSample;
import ai.neuronix.loss.MeanSquaredErrorLoss;
import ai.neuronix.math.Matrix;
import ai.neuronix.math.MatrixFactory;
import ai.neuronix.nn.DenseLayer;
import ai.neuronix.nn.NeuralNetwork;
import ai.neuronix.nn.ReLULayer;
import ai.neuronix.optimizer.SGDOptimizer;
import ai.neuronix.training.Trainer;
import java.util.List;
import org.junit.jupiter.api.Test;

public class NeuralNetworkTrainingTest {

  @Test
  void shouldLearnSimpleAddition() {

    NeuralNetwork network = new NeuralNetwork()
        .add(new DenseLayer(2, 1));

    Trainer trainer = new Trainer(
        network,
        new MeanSquaredErrorLoss(),
        new SGDOptimizer(0.01));

    List<TrainingSample> dataset = List.of(
        new TrainingSample(
            matrix(1, 1),
            matrix(2)));

    double before = new MeanSquaredErrorLoss().calculate(
        network.forward(matrix(1, 1)),
        matrix(2));

    trainer.train(dataset, 1000);

    double after = new MeanSquaredErrorLoss().calculate(
        network.forward(matrix(1, 1)),
        matrix(2));

    assertThat(after).isLessThan(before);
  }

  @Test
  void shouldReduceTrainingLoss() {

    NeuralNetwork network = new NeuralNetwork()
        .add(new DenseLayer(
            matrix(new double[][] {
                {0.10, -0.20},
                {0.30,  0.40},
                {-0.50, 0.20},
                {0.10,  0.60}
            }),
            MatrixFactory.zeros(4, 1)))
        .add(new ReLULayer())
        .add(new DenseLayer(
            matrix(new double[][] {
                {0.20, -0.30, 0.40, 0.10}
            }),
            MatrixFactory.zeros(1, 1)));

    Trainer trainer = new Trainer(
        network,
        new MeanSquaredErrorLoss(),
        new SGDOptimizer(0.01));

    List<TrainingSample> dataset = List.of(
        sample(1, 1, 2),
        sample(2, 3, 5),
        sample(3, 4, 7),
        sample(5, 7, 12),
        sample(8, 2, 10),
        sample(4, 6, 10)
    );

    double before = averageLoss(network, dataset);

    trainer.train(dataset, 2000);

    double after = averageLoss(network, dataset);

    assertThat(after).isLessThan(before);
  }

  private static TrainingSample sample(double x1, double x2, double y) {

    return new TrainingSample(matrix(x1, x2), matrix(y));
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

  private double averageLoss(NeuralNetwork network, List<TrainingSample> dataset) {

    MeanSquaredErrorLoss loss = new MeanSquaredErrorLoss();

    double sum = 0;

    for (TrainingSample sample : dataset) {
      sum += loss.calculate(network.forward(sample.input()), sample.expected());
    }

    return sum / dataset.size();
  }
}
