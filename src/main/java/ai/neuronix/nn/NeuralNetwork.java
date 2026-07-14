package ai.neuronix.nn;

import ai.neuronix.math.Matrix;
import ai.neuronix.optimizer.Parameter;
import ai.neuronix.optimizer.Parameterized;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

public class NeuralNetwork implements Layer {

  private final List<Layer> layers = new ArrayList<>();

  public NeuralNetwork add(Layer layer) {
    layers.add(layer);
    return this;
  }

  @Override
  public Matrix forward(Matrix input) {

    Matrix output = input;

    for (Layer layer : layers) {
      output = layer.forward(output);
    }

    return output;
  }

  @Override
  public Matrix backward(Matrix gradient) {

    ListIterator<Layer> iterator = layers.listIterator(layers.size());

    while (iterator.hasPrevious()) {
      gradient = iterator.previous().backward(gradient);
    }

    return gradient;
  }

  public List<Parameter> parameters() {

    List<Parameter> parameters = new ArrayList<>();

    for (Layer layer : layers) {
      if (layer instanceof Parameterized parameterized) {
        parameters.addAll(parameterized.parameters());
      }
    }

    return parameters;
  }
}
