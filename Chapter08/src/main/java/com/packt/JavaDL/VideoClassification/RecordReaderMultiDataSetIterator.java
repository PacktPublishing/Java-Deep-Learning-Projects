package com.packt.JavaDL.VideoClassification;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.datavec.api.records.metadata.RecordMetaDataComposable;
import org.datavec.api.records.metadata.RecordMetaDataComposableMap;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator.AlignmentMode;
import org.deeplearning4j.datasets.datavec.exception.ZeroLengthSequenceException;
import org.jcodec.api.FrameGrab;
import org.jcodec.api.JCodecException;
import org.jcodec.common.io.FileChannelWrapper;
import org.jcodec.common.model.Picture;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

// based on org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIterator.java
public class RecordReaderMultiDataSetIterator {
    private static final String READER_KEY = "reader";
    private static final String READER_KEY_LABEL = "reader_labels";
    private final AlignmentMode alignmentMode = AlignmentMode.EQUAL_LENGTH; 
    private final boolean timeSeriesRandomOffset = false;
    private final ArrayList<SubsetDetails> inputs = new ArrayList<>();
    private final ArrayList<SubsetDetails> outputs = new ArrayList<>();
    private DataSetPreProcessor preProcessor;
    private final boolean collectMetaData = false;
    private final boolean singleSequenceReaderMode = false;
    private final boolean underlyingIsDisjoint = false;
    private int totalOutcomes = -1;
    private final ImageLoader imageLoader;

    public RecordReaderMultiDataSetIterator(int classCount, ImageLoader imageLoader) {
        inputs.add(new SubsetDetails(READER_KEY, true, false, -1, -1, -1));
        outputs.add(new SubsetDetails(READER_KEY_LABEL, false, true, classCount, 0, -1));
        this.imageLoader = imageLoader;
    }

    // my function
    public DataSet nextDataSet(FileChannelWrapper in, int framesTotal, FrameGrab fg, Integer label, int labelCount) throws IOException, JCodecException {
        try {
            List<List<List<Writable>>> features = new ArrayList<>();
            List<List<List<Writable>>> labels = new ArrayList<>();

            List<List<Writable>> batchElem = new ArrayList<>();
            List<List<Writable>> labelBatchElem = new ArrayList<>();
            for (int frame = 0; frame < framesTotal; frame++) {
                List<Writable> frames = new ArrayList<>();
                List<Writable> labelFrames = new ArrayList<>();

                //long startFrame = System.nanoTime();
                Picture picture = fg.getNativeFrame();
                //INDArray array0 = imageLoader.toRaveledTensor(AWTUtil.toBufferedImage(picture));
                //INDArray array = Nd4j.create(byte2dtoFloat1d(picture.getData()));
                INDArray array = imageLoader.toRaveledTensor(AWTUtil.toBufferedImage(picture));
                frames.add(new NDArrayWritable(array));
                labelFrames.add(new Text(label.toString()));
                //System.out.println(" Frame " + frame + " readed from " + ((System.nanoTime() - startFrame) / 1000000) + " ms");

                batchElem.add(frames);
                labelBatchElem.add(labelFrames);
            }


            features.add(batchElem);
            labels.add(labelBatchElem);

            HashMap<String, List<List<List<Writable>>>> nextSeqRRVals = new HashMap<>();
            nextSeqRRVals.put(READER_KEY, features);
            nextSeqRRVals.put(READER_KEY_LABEL, labels);
            MultiDataSet mds = nextMultiDataSet(Collections.emptyMap(), null, nextSeqRRVals, null);
            DataSet ds = mdsToDataSet(mds);

            if (totalOutcomes == -1) {
                ds.getFeatures().size(1);
                totalOutcomes = ds.getLabels().size(1);
            }
            return ds;
        } finally {
            in.close();
        }
    }

    public MultiDataSet nextMultiDataSet(Map<String, List<List<Writable>>> nextRRVals,
                                         Map<String, List<INDArray>> nextRRValsBatched,
                                         Map<String, List<List<List<Writable>>>> nextSeqRRVals,
                                         List<RecordMetaDataComposableMap> nextMetas) {
        int minExamples = Integer.MAX_VALUE;
        for (List<List<Writable>> exampleData : nextRRVals.values()) {
            minExamples = Math.min(minExamples, exampleData.size());
        }
        if (nextRRValsBatched != null) {
            for (List<INDArray> exampleData : nextRRValsBatched.values()) {
                //Assume all NDArrayWritables here
                for (INDArray w : exampleData) {
                    int n = w.size(0);
                    minExamples = Math.min(minExamples, n);
                }
            }
        }
        for (List<List<List<Writable>>> exampleData : nextSeqRRVals.values()) {
            minExamples = Math.min(minExamples, exampleData.size());
        }

        if (minExamples == Integer.MAX_VALUE)
            throw new RuntimeException("Error occurred during data set generation: no readers?"); //Should never happen

        //In order to align data at the end (for each example individually), we need to know the length of the
        // longest time series for each example
        int[] longestSequence = null;
        int longestTS = -1;

        long rngSeed = /*(timeSeriesRandomOffset ? timeSeriesRandomOffsetRng.nextLong() : -1);*/ -1;
        int inputsSize = inputs.size();
        int outputsSize = outputs.size();
        
        Pair<INDArray[], INDArray[]> features = convertFeaturesOrLabels(new INDArray[inputsSize],
                new INDArray[inputsSize], inputs, minExamples, nextRRVals, nextRRValsBatched, nextSeqRRVals,
                longestTS, longestSequence, rngSeed);

        //Third: create the outputs/labels
        Pair<INDArray[], INDArray[]> labels = convertFeaturesOrLabels(new INDArray[outputsSize],
                new INDArray[outputsSize], outputs, minExamples, nextRRVals, nextRRValsBatched,
                nextSeqRRVals, longestTS, longestSequence, rngSeed);

        MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(features.getFirst(), labels.getFirst(),
                features.getSecond(), labels.getSecond());
        return mds;
    }

    private Pair<INDArray[], INDArray[]> convertFeaturesOrLabels(INDArray[] featuresOrLabels, INDArray[] masks,
                                                                 List<SubsetDetails> subsetDetails,
                                                                 int minExamples, Map<String, List<List<Writable>>> nextRRVals,
                                                                 Map<String, List<INDArray>> nextRRValsBatched,
                                                                 Map<String, List<List<List<Writable>>>> nextSeqRRVals, 
                                                                 int longestTS, int[] longestSequence,
                                                                 long rngSeed) {
        boolean hasMasks = false;
        int i = 0;

        for (SubsetDetails d : subsetDetails) {
            List<List<List<Writable>>> list = nextSeqRRVals.get(d.readerName);
            Pair<INDArray, INDArray> p =
                    convertWritablesSequence(list, minExamples, longestTS, d, longestSequence, rngSeed);
            featuresOrLabels[i] = p.getFirst();
            masks[i] = p.getSecond();
            if (masks[i] != null)
                hasMasks = true;
            i++;
        }

        return new Pair<>(featuresOrLabels, hasMasks ? masks : null);
    }

    private Pair<INDArray, INDArray> convertWritablesSequence(List<List<List<Writable>>> list, int minValues,
                                                              int maxTSLength, SubsetDetails details, int[] longestSequence, long rngSeed) {
        if (maxTSLength == -1)
            maxTSLength = list.get(0).size();
        INDArray arr;

        if (list.get(0).isEmpty()) {
            throw new ZeroLengthSequenceException("Zero length sequence encountered");
        }

        List<Writable> firstStep = list.get(0).get(0);

        int size = 0;
        if (details.entireReader) {
            //Need to account for NDArrayWritables etc in list:
            for (Writable w : firstStep) {
                if (w instanceof NDArrayWritable) {
                    size += ((NDArrayWritable) w).get().size(1);
                } else {
                    size++;
                }
            }
        } else if (details.oneHot) {
            size = details.oneHotNumClasses;
        } else {
            //Need to account for NDArrayWritables etc in list:
            for (int i = details.subsetStart; i <= details.subsetEndInclusive; i++) {
                Writable w = firstStep.get(i);
                if (w instanceof NDArrayWritable) {
                    size += ((NDArrayWritable) w).get().size(1);
                } else {
                    size++;
                }
            }
        }
        arr = Nd4j.create(new int[] {minValues, size, maxTSLength}, 'f');

        boolean needMaskArray = false;
        for (List<List<Writable>> c : list) {
            if (c.size() < maxTSLength)
                needMaskArray = true;
        }

        if (needMaskArray && alignmentMode == org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator.AlignmentMode.EQUAL_LENGTH) {
            throw new UnsupportedOperationException(
                    "Alignment mode is set to EQUAL_LENGTH but variable length data was "
                            + "encountered. Use AlignmentMode.ALIGN_START or AlignmentMode.ALIGN_END with variable length data");
        }

        INDArray maskArray;
        if (needMaskArray) {
            maskArray = Nd4j.ones(minValues, maxTSLength);
        } else {
            maskArray = null;
        }

        //Don't use the global RNG as we need repeatability for each subset (i.e., features and labels must be aligned)
        Random rng = null;
        if (timeSeriesRandomOffset) {
            rng = new Random(rngSeed);
        }

        for (int i = 0; i < minValues; i++) {
            List<List<Writable>> sequence = list.get(i);

            //Offset for alignment:
            int startOffset;
            if (alignmentMode == org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_START || alignmentMode == org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator.AlignmentMode.EQUAL_LENGTH) {
                startOffset = 0;
            } else {
                //Align end
                //Only practical differences here are: (a) offset, and (b) masking
                startOffset = longestSequence[i] - sequence.size();
            }

            if (timeSeriesRandomOffset) {
                int maxPossible = maxTSLength - sequence.size() + 1;
                startOffset = rng.nextInt(maxPossible);
            }

            int t = 0;
            int k;
            for (List<Writable> timeStep : sequence) {
                k = startOffset + t++;

                if (details.entireReader) {
                    //Convert entire reader contents, without modification
                    Iterator<Writable> iter = timeStep.iterator();
                    int j = 0;
                    while (iter.hasNext()) {
                        Writable w = iter.next();

                        if (w instanceof NDArrayWritable) {
                            INDArray row = ((NDArrayWritable) w).get();

                            arr.put(new INDArrayIndex[] {NDArrayIndex.point(i),
                                    NDArrayIndex.interval(j, j + row.length()), NDArrayIndex.point(k)}, row);
                            j += row.length();
                        } else {
                            arr.putScalar(i, j, k, w.toDouble());
                            j++;
                        }
                    }
                } else if (details.oneHot) {
                    //Convert a single column to a one-hot representation
                    Writable w = null;
                    if (timeStep instanceof List)
                        w = timeStep.get(details.subsetStart);
                    else {
                        Iterator<Writable> iter = timeStep.iterator();
                        for (int x = 0; x <= details.subsetStart; x++)
                            w = iter.next();
                    }
                    int classIdx = w.toInt();
                    if (classIdx >= details.oneHotNumClasses) {
                        throw new IllegalStateException("Cannot convert sequence writables to one-hot: class index " + classIdx
                                + " >= numClass (" + details.oneHotNumClasses + "). (Note that classes are zero-" +
                                "indexed, thus only values 0 to nClasses-1 are valid)");
                    }
                    arr.putScalar(i, classIdx, k, 1.0);
                } else {
                    //Convert a subset of the columns...
                    int l = 0;
                    for (int j = details.subsetStart; j <= details.subsetEndInclusive; j++) {
                        Writable w = timeStep.get(j);

                        if (w instanceof NDArrayWritable) {
                            INDArray row = ((NDArrayWritable) w).get();
                            arr.put(new INDArrayIndex[] {NDArrayIndex.point(i),
                                    NDArrayIndex.interval(l, l + row.length()), NDArrayIndex.point(k)}, row);

                            l += row.length();
                        } else {
                            arr.putScalar(i, l++, k, w.toDouble());
                        }
                    }
                }
            }

            //For any remaining time steps: set mask array to 0 (just padding)
            if (needMaskArray) {
                //Masking array entries at start (for align end)
                if (timeSeriesRandomOffset || alignmentMode == org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END) {
                    for (int t2 = 0; t2 < startOffset; t2++) {
                        maskArray.putScalar(i, t2, 0.0);
                    }
                }

                //Masking array entries at end (for align start)
                int lastStep = startOffset + sequence.size();
                if (timeSeriesRandomOffset || alignmentMode == org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_START || lastStep < maxTSLength) {
                    for (int t2 = lastStep; t2 < maxTSLength; t2++) {
                        maskArray.putScalar(i, t2, 0.0);
                    }
                }
            }
        }

        return new Pair<>(arr, maskArray);
    }

    private DataSet mdsToDataSet(MultiDataSet mds) {
        INDArray f;
        INDArray fm;
        if (underlyingIsDisjoint) {
            //Rare case: 2 input arrays -> concat
            INDArray f1 = getOrNull(mds.getFeatures(), 0);
            INDArray f2 = getOrNull(mds.getFeatures(), 1);
            fm = getOrNull(mds.getFeaturesMaskArrays(), 0); //Per-example masking only on the input -> same for both

            //Can assume 3d features here
            f = Nd4j.createUninitialized(new int[] {f1.size(0), f1.size(1) + f2.size(1), f1.size(2)});
            f.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(0, f1.size(1)), NDArrayIndex.all()},
                    f1);
            f.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(f1.size(1), f1.size(1) + f2.size(1)),
                    NDArrayIndex.all()}, f2);
        } else {
            //Standard case
            f = getOrNull(mds.getFeatures(), 0);
            fm = getOrNull(mds.getFeaturesMaskArrays(), 0);
        }

        INDArray l = getOrNull(mds.getLabels(), 0);
        INDArray lm = getOrNull(mds.getLabelsMaskArrays(), 0);

        DataSet ds = new DataSet(f, l, fm, lm);

        if (collectMetaData) {
            List<Serializable> temp = mds.getExampleMetaData();
            List<Serializable> temp2 = new ArrayList<>(temp.size());
            for (Serializable s : temp) {
                RecordMetaDataComposableMap m = (RecordMetaDataComposableMap) s;
                if (singleSequenceReaderMode) {
                    temp2.add(m.getMeta().get(READER_KEY));
                } else {
                    RecordMetaDataComposable c = new RecordMetaDataComposable(m.getMeta().get(READER_KEY),
                            m.getMeta().get(READER_KEY_LABEL));
                    temp2.add(c);
                }
            }
            ds.setExampleMetaData(temp2);
        }

        if (preProcessor != null) {
            preProcessor.preProcess(ds);
        }

        return ds;
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    private INDArray getOrNull(INDArray[] arr, int idx) {
        if (arr == null || arr.length == 0) {
            return null;
        }
        return arr[idx];
    }

    private static class SubsetDetails implements Serializable {
        /**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		private final String readerName;
        private final boolean entireReader;
        private final boolean oneHot;
        private final int oneHotNumClasses;
        private final int subsetStart;
        private final int subsetEndInclusive;

        private SubsetDetails(String readerName, boolean entireReader, boolean oneHot, int oneHotNumClasses, int subsetStart, int subsetEndInclusive) {
            this.readerName = readerName;
            this.entireReader = entireReader;
            this.oneHot = oneHot;
            this.oneHotNumClasses = oneHotNumClasses;
            this.subsetStart = subsetStart;
            this.subsetEndInclusive = subsetEndInclusive;
        }
    }
}
