package com.packt.JavaDL.VideoClassification;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class UCF101Reader {
    public static final int V_WIDTH = 320;
    public static final int V_HEIGHT = 240;
    public static final int V_NFRAMES = 100;
    private final String dataDirectory;
    private volatile Map<Integer, String> _labelMap;

    public UCF101Reader(String dataDirectory) {
        this.dataDirectory = dataDirectory.endsWith("/") ? dataDirectory : dataDirectory + "/";
    }

    public DataSetIterator getDataSetIterator(int startIdx, int nExamples, int miniBatchSize) throws Exception {
        ExistingDataSetIterator iter = new ExistingDataSetIterator(createDataSetIterable(startIdx, nExamples, miniBatchSize));
        return new AsyncDataSetIterator(iter,1);
    }

    private UCF101RecordIterable createDataSetIterable(int startIdx, int nExamples, int miniBatchSize) throws IOException {
        return new UCF101RecordIterable(dataDirectory, labelMap(), V_WIDTH, V_HEIGHT, startIdx, nExamples);
    }

    public Map<Integer, String> labelMap() throws IOException {
        if (_labelMap == null) {
            synchronized (this) {
                if (_labelMap == null) {
                    File root = new File(dataDirectory);
                    _labelMap = Files.list(root.toPath()).map(f -> f.getFileName().toString())
                            .sorted().collect(HashMap::new, (h, f) -> h.put(h.size(), f), (h, o) -> {});
                }
            }
        }
        return _labelMap;
    }

    public int fileCount() {
        return (int) UCF101RecordIterable.rowsStream(dataDirectory).count();
    }
}
