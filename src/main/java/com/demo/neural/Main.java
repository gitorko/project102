package com.demo.neural;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.cpu.nativecpu.buffer.FloatBuffer;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@Slf4j
@SpringBootApplication
public class Main {

    //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
    final int LABEL_INDEX = 4;
    //3 classes (types of iris flowers) in the iris data set. Classes have integer values
    // 0, 1 or 2
    final int NUM_CLASS = 3;
    //Iris data set: 150 examples total. We are loading all of them into one DataSet
    // (not recommended for large data sets)
    final int BATCH_SIZE = 150;

    final List<String> flowerType = Arrays.asList("Iris Setosa", "Iris Versicolour", "Iris Virginica");

    public static void main(String[] args) {
        SpringApplication.run(Main.class, args);
    }

    @Bean
    public CommandLineRunner sendData() {
        return args -> {
            generateModel();
            log.info("Flower type is {}", predictForInput(new float[]{5.1f, 3.5f, 1.4f, 0.2f}));
            log.info("Flower type is {}", predictForInput(new float[]{6.5f, 3.0f, 5.5f, 1.8f}));
        };

    }

    private DataSet loadDataSet() throws IOException, InterruptedException {
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new File("src/main/resources/iris.txt")));

        //RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, LABEL_INDEX, NUM_CLASS);
        DataSet allData = iterator.next();
        allData.shuffle();
        return allData;
    }

    private void generateModel() throws IOException, InterruptedException {
        DataSet allData = loadDataSet();
        //Use 65% of data for training
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.fit(trainingData);
        //Apply normalization to the training data
        normalizer.transform(trainingData);
        //Apply normalization to the test data. This is using statistics calculated from the *training* set
        normalizer.transform(testData);

        final int numInputs = 4;
        int outputNum = 3;
        long seed = 6;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .build())
                .layer(new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                        .nIn(3).nOut(outputNum).build())
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < 1000; i++) {
            model.fit(trainingData);
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
        saveModelAndNormalizer(model, normalizer);

    }

    private void saveModelAndNormalizer(MultiLayerNetwork model, DataNormalization normalizer) throws IOException {
        log.info("Saving model & normalizer!");
        File modelFile = new File("model.file");
        model.save(modelFile, false);

        File normalizerFile = new File("normalize.file");
        NormalizerSerializer.getDefault().write(normalizer, normalizerFile);
    }

    private String predictForInput(float[] input) throws Exception {
        log.info("Loading model & normalizer!");
        File modelFile = new File("model.file");
        MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, false);
        File normalizerFile = new File("normalize.file");
        DataNormalization normalizer = NormalizerSerializer.getDefault().restore(normalizerFile);

        DataBuffer dataBuffer = new FloatBuffer(input);
        NDArray ndArray = new NDArray(1, 4);
        ndArray.setData(dataBuffer);

        normalizer.transform(ndArray);
        INDArray result = model.output(ndArray, false);
        getIndexLabel(result);

        return flowerType.get(getIndexLabel(result));
    }

    private int getIndexLabel(INDArray predictions) {
        int maxIndex = 0;
        for (int i = 0; i < 3; i++) {
            if (predictions.getFloat(i) > predictions.getFloat(maxIndex)) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}
