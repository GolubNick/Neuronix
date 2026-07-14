package ai.neuronix.dataset.loader;

import ai.neuronix.dataset.Dataset;
import java.io.IOException;
import java.nio.file.Path;

public interface DatasetLoader {

  Dataset load(Path path) throws IOException;
}
