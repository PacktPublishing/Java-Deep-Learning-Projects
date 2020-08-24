package com.packt.JavaDL.VideoClassification;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.apache.commons.lang3.tuple.Pair;
import org.datavec.image.loader.ImageLoader;
import org.jcodec.api.FrameGrab;
import org.jcodec.api.JCodecException;
import org.jcodec.api.specific.AVCMP4Adaptor;
import org.jcodec.common.SeekableDemuxerTrack;
import org.jcodec.common.io.FileChannelWrapper;
import org.jcodec.common.io.NIOUtils;
import org.jcodec.containers.mp4.demuxer.MP4Demuxer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.DataSet;

public class UCF101RecordIterable implements Iterable<DataSet> {
    private final String dataDirectory;
    private final Map<Integer, String> labelMap;
    private final Map<String, Integer> labelMapInversed;
    private final ImageLoader imageLoader;
    private final RecordReaderMultiDataSetIterator recordReaderMultiDataSetIterator;
    private final int skip;
    private final int limit;

    public UCF101RecordIterable(String dataDirectory, Map<Integer, String> labelMap, int rows, int cols, int skip, int limit) {
        this.dataDirectory = dataDirectory;
        this.labelMap = labelMap;
        this.labelMapInversed = new HashMap<>();
        for (Map.Entry<Integer, String> e : labelMap.entrySet()) {
            labelMapInversed.put(e.getValue(), e.getKey());
        }
        imageLoader = new ImageLoader(rows, cols);
        labelMap.size();
        recordReaderMultiDataSetIterator = new RecordReaderMultiDataSetIterator(labelMap.size(), imageLoader);
        this.skip = skip;
        this.limit = limit;
    }

    @NotNull
    @Override
    public Iterator<DataSet> iterator() {
        return rowsStream(dataDirectory).skip(this.skip).limit(this.limit).flatMap(p -> dataSetsStreamFromFile(p.getKey(), p.getValue())).iterator();
    }

    public static Stream<Pair<Path, String>> rowsStream(String dataDirectory) {
        try {
            List<Pair<Path, String>> files = Files.list(Paths.get(dataDirectory)).flatMap(dir -> {
                try {
                    return Files.list(dir).map(p -> Pair.of(p, dir.getFileName().toString()));
                } catch (IOException e) {
                    e.printStackTrace();
                    return Stream.empty();
                }
            }).collect(Collectors.toList());
            Collections.shuffle(files, new Random(43));
            return files.stream();
        } catch (IOException e) {
            e.printStackTrace();
            return Stream.empty();
        }
    }

    private Stream<DataSet> dataSetsStreamFromFile(Path path, String label) {
        // return Stream.of(dataSetsIteratorFromFile(path, label).next());
        return StreamSupport.stream(Spliterators.spliteratorUnknownSize(dataSetsIteratorFromFile(path, label), Spliterator.ORDERED), false);
    }

    private Iterator<DataSet> dataSetsIteratorFromFile(Path path, String label) {
        FileChannelWrapper _in = null;
        try {
            _in = NIOUtils.readableChannel(path.toFile());
            MP4Demuxer d1 = MP4Demuxer.createMP4Demuxer(_in);
            SeekableDemuxerTrack videoTrack_ = (SeekableDemuxerTrack)d1.getVideoTrack();
            FrameGrab fg = new FrameGrab(videoTrack_, new AVCMP4Adaptor(videoTrack_.getMeta()));
            //fg.seekToFramePrecise(frameNumber).getNativeFrame();
            final int framesTotal = videoTrack_.getMeta().getTotalFrames();
            return Collections.singleton(recordReaderMultiDataSetIterator.nextDataSet(_in, framesTotal, fg, labelMapInversed.get(label), labelMap.size())).iterator();
            //return new FrameIterator(framesTotal, _in, fg, labelArray, imageLoader);
        } catch (IOException | JCodecException e) {
            e.printStackTrace();
            return Collections.emptyIterator();
        }
    }
}
