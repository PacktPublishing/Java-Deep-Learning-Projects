package com.packt.JavaDL.VideoClassification;

import org.apache.commons.lang3.tuple.Pair;
import org.jcodec.api.FrameGrab;
import org.jcodec.api.JCodecException;
import org.jcodec.common.model.Picture;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class JCodecTest {
    private String dataDirectory = "C:/Users/admin-karim/Desktop/VideoData/UCF101_MP4/";

    public static void main(String[] args) throws IOException, JCodecException {
        JCodecTest test = new JCodecTest();
        test.testReadFrame(new FxShow());
    }

    private Stream<Pair<Path, String>> rowsStream() {
        try {
            return Files.list(Paths.get(dataDirectory)).flatMap(dir -> {
                try {
                    return Files.list(dir).map(p -> Pair.of(p, dir.getFileName().toString()));
                } catch (IOException e) {
                    e.printStackTrace();
                    return Stream.empty();
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
            return Stream.empty();
        }
    }

    private void testReadFrame(Consumer<Picture> consumer) throws IOException, JCodecException {
        next:
        for (Iterator<Pair<Path, String>> iter = rowsStream().iterator(); iter.hasNext(); ) {
            Pair<Path, String> pair = iter.next();
            Path path = pair.getKey();
            pair.getValue();
            for (int i = 0; i < 100; i++) {
                try {
                    Picture picture = FrameGrab.getFrameFromFile(path.toFile(), i);
                    consumer.accept(picture);
                } catch (Throwable ex) {
                    System.out.println(ex.toString() + " frame " + i + " " + path.toString());
                    continue next;
                }
            }
            System.out.println("OK " + path.toString());
        }
    }

    @SuppressWarnings("serial")
	public static class FxShow extends JPanel implements Consumer<Picture> {

        private JFrame frame;
        private Picture lastPicture;

        public FxShow() {
            frame = new JFrame();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.getContentPane().add(this);
            frame.setSize(320, 240);
            frame.setVisible(true);
        }

        @Override
        public void accept(Picture picture) {
            lastPicture = picture;
            frame.repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            if (lastPicture != null) {
                g.clearRect(0, 0, lastPicture.getWidth(), lastPicture.getHeight());
                g.drawImage(AWTUtil.toBufferedImage(lastPicture), 0, 0, this);
            } else {
                super.paintComponent(g);
            }
        }
    }
}
