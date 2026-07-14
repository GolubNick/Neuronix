package ai.neuronix.optimizer;

import java.util.List;

public interface Optimizer {
    void step(List<Parameter> parameters);
}
