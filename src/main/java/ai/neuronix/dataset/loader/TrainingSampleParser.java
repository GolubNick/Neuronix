package ai.neuronix.dataset.loader;

import ai.neuronix.dataset.TrainingSample;

public interface TrainingSampleParser {
  TrainingSample parse(String line);
}
