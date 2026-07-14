package ai.neuronix.dataset;

public interface Dataset {

  int size();

  TrainingSample get(int index);
}
