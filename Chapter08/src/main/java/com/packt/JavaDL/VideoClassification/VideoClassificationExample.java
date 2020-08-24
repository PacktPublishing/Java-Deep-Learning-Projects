package com.packt.JavaDL.VideoClassification;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Example: Combine convolutional, max pooling, dense (feed forward) and recurrent (LSTM) layers to classify each frame of a video
 * @author Md. Rezaul Karim
 */
public class VideoClassificationExample {
	private static MultiLayerConfiguration conf;
	private static MultiLayerNetwork net;
	private static String modelPath = "bin/ConvLSTM_Model.zip";
	private static int NUM_CLASSES;
	private static int nTrainEpochs = 1;
	private static int miniBatchSize = 10;
	private static int NUM_EXAMPLE = 10;
	
    public static void main(String[] args) throws Exception {        
        String dataDirectory = "C:/Users/admin-karim/Desktop/VideoData/UCF101_MP4/";// Paths.get("data", "UCF-101-mp4").toAbsolutePath().toString();
        UCF101Reader reader = new UCF101Reader(dataDirectory); 
        NUM_CLASSES = reader.labelMap().size();        
        
        int examplesOffset = 0; // start from N-th file
        int nExamples = Math.min(NUM_EXAMPLE, reader.fileCount()); // use only "nExamples" for train/test
        int testStartIdx = examplesOffset + Math.max(2, (int) (0.9 * nExamples));  //90% in train, 10% in test
        int nTest = nExamples - testStartIdx + examplesOffset;
        System.out.println("Dataset consist of " + reader.fileCount() + " images, use " + nExamples + " of them");        

        //Conduct learning
        System.out.println("Starting training...");       
        DataSetIterator trainData = reader.getDataSetIterator(examplesOffset, nExamples - nTest, miniBatchSize);        
        networkTrainer(reader, trainData);
        
        //Save network and video configuration
        saveConfigs();
        
        //Save the trained model
        saveNetwork();

        //Evaluate classification performance:
        System.out.println("Use " + String.valueOf(nTest) + " images for validation");
        DataSetIterator testData = reader.getDataSetIterator(testStartIdx, nExamples, miniBatchSize);
        evaluateClassificationPerformance(net,testStartIdx,nTest, testData);        
    }
    
	private static void networkTrainer(UCF101Reader reader, DataSetIterator trainData) throws Exception {        
    	 //Set up network architecture:
        conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.001) //l2 regularization on all layers
                .updater(new Adam(0.001))
                .list()
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(3) //3 channels: RGB
                        .nOut(30)
                        .stride(4, 4)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.RELU)
                        .build())   //Output: (130-10+0)/4+1 = 31 -> 31*31*30
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2).build())   //(31-3+0)/2+1 = 15
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nIn(30)
                        .nOut(10)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.RELU)
                        .build())   //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
                .layer(3, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(2340) // 13 * 18 * 10 = 2340, see CNN layer width x height
                        .nOut(50)
                        .weightInit(WeightInit.RELU)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .updater(new AdaGrad(0.01))
                        .build())
                .layer(4, new LSTM.Builder()
                        .activation(Activation.SOFTSIGN)
                        .nIn(50)
                        .nOut(50)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new AdaGrad(0.008))
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(50)
                        .nOut(NUM_CLASSES)    
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .inputPreProcessor(0, new RnnToCnnPreProcessor(UCF101Reader.V_HEIGHT, UCF101Reader.V_WIDTH, 3))
                .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(13, 18, 10))
                .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
                .pretrain(false).backprop(true)
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(UCF101Reader.V_NFRAMES / 5)
                .tBPTTBackwardLength(UCF101Reader.V_NFRAMES / 5)
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        System.out.println("Number of parameters in network: " + net.numParams());
        for( int i=0; i<net.getnLayers(); i++ ){
            System.out.println("Layer " + i + " nParams = " + net.getLayer(i).numParams());
        }
        
      //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored.Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage(); 

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        
        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));
        
        for (int i = 0; i < nTrainEpochs; i++) {
            int j = 0;
            while(trainData.hasNext()) {
                long start = System.nanoTime();
                DataSet example = trainData.next();
                net.fit(example);
                System.out.println(" Example " + j + " processed in " + ((System.nanoTime() - start) / 1000000) + " ms");
                j++;
            }
            System.out.println("Epoch " + i + " complete");
        }
    }
    
    private static void saveConfigs() throws IOException {
        Nd4j.saveBinary(net.params(),new File("videomodel.bin"));
        FileUtils.writeStringToFile(new File("videoconf.json"), conf.toJson());
    }
    
    private static void saveNetwork() throws IOException {        
		//Save the model
        File locationToSave = new File(modelPath);      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                     //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);     
    }

    private static void evaluateClassificationPerformance(MultiLayerNetwork net, int testStartIdx, int nExamples, DataSetIterator testData) throws Exception {
        //Assuming here that the full test data set doesn't fit in memory -> load 10 examples at a time
        Evaluation evaluation = new Evaluation(NUM_CLASSES);

        while(testData.hasNext()) {
            DataSet dsTest = testData.next();
            INDArray predicted = net.output(dsTest.getFeatureMatrix(), false);
            INDArray actual = dsTest.getLabels();
            evaluation.evalTimeSeries(actual, predicted);
        }

        System.out.println(evaluation.stats());
    }
}
