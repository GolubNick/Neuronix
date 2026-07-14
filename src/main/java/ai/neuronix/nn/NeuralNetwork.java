package ai.neuronix.nn;

import ai.neuronix.math.Matrix;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

public class NeuralNetwork implements Layer, Trainable {

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

    @Override
    public void update(double learningRate) {

        for (Layer layer : layers) {

            if (layer instanceof Trainable trainable) {
                trainable.update(learningRate);
            }
        }
    }
}
