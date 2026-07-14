package ai.neuronix.dataset;

import java.util.List;
import java.util.Objects;

public class ListDataset implements Dataset {

  private final List<TrainingSample> samples;

  public ListDataset(List<TrainingSample> samples) {
    this.samples = List.copyOf(Objects.requireNonNull(samples));

    if (samples.isEmpty()) {
      throw new IllegalArgumentException("Dataset must not be empty.");
    }
  }

  @Override
  public int size() {
    return samples.size();
  }

  @Override
  public TrainingSample get(int index) {
    return samples.get(index);
  }
}
