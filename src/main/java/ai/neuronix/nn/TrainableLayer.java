package ai.neuronix.nn;

public interface TrainableLayer extends Layer {
  void update(double learningRate);
}
