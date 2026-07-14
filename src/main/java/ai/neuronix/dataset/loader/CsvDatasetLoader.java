package ai.neuronix.dataset.loader;

import ai.neuronix.dataset.Dataset;
import ai.neuronix.dataset.ListDataset;
import ai.neuronix.dataset.TrainingSample;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class CsvDatasetLoader implements DatasetLoader {

  private final TrainingSampleParser parser;

  public CsvDatasetLoader(TrainingSampleParser parser) {

    this.parser = parser;
  }

  @Override
  public Dataset load(Path path) throws IOException {

    List<TrainingSample> samples =
        Files.lines(path).filter(line -> !line.isBlank()).map(parser::parse).toList();

    return new ListDataset(samples);
  }
}
