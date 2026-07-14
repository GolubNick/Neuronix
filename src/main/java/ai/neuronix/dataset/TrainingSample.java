package ai.neuronix.dataset;

import ai.neuronix.math.Matrix;

public record TrainingSample(Matrix input, Matrix expected) {}
