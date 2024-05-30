import java.io.*;
import java.util.*;

/*
 * 
 * This class implements an improved k-Nearest Neighbors (kNN) algorithm.
 * The methods in this class include:
 * 1. loadDataLabels: loads data and labels from files
 * 2. normalizeData: normalizes the data
 * 3. calculatingManhattanDistance: calculates the manhattan distance
 * 4. findIndexOfLargestDistance: finds the index of the largest distance in the array
 * 5. checkNearestNeighbours: finds the k nearest neighbors of a test instance
 * 6. getPredictedLabelWithImprovedApproach: finds the predicted label of a test instance
 * 7. outputToFile: writes the predicted labels to a file
 * 8. main: main method
 * 
 * @author: SS2697
 * @version: 1.0
 * @date: 2023/12/06
 * 
 */
public class kNN2 {

    /**
     * The main method for executing kNN prediction with the improved approach.
     * Loads data, normalizes it, performs kNN prediction with k = 7 using Manhattan distance,
     * calculates accuracy, and outputs results.
     *
     * @param args Command-line arguments.
     */
    public static void main(String[] args) {
        int FEATURE_SIZE = 61;
        int TRAIN_SIZE = 200;
        int TEST_SIZE = 200;
        
        int[] predictedLabels = new int[TEST_SIZE];
        int[] testLabels = new int[TEST_SIZE];
        double[][] test = new double[TEST_SIZE][FEATURE_SIZE];
        double[][] train = new double[TRAIN_SIZE][FEATURE_SIZE];
        int[] trainLabel = new int[TRAIN_SIZE];
        int k = 7; 
        int correct = 0;

        loadDataLabels("train_data.txt", train, "train_label.txt", trainLabel);
        loadDataLabels("test_data.txt", test, "test_label.txt", testLabels);
        normalizeData(train);
        normalizeData(test);

        // Perform kNN prediction with modified approach (Manhattan distance)
        // K is set to 7
        for (int i = 0; i < TEST_SIZE; i++) {
            predictedLabels[i] = getPredictedLabelWithImprovedApproach(test[i], train, trainLabel, k);
        }

        // Calculate accuracy and display it
        for (int i = 0; i < TEST_SIZE; i++) {
            if (predictedLabels[i] == testLabels[i]) {
                correct++;
            }
        }
        double accuracy = (double) correct / TEST_SIZE * 100;
        System.out.println("Accuracy: " + accuracy + "%");

        // Write predicted labels to output2.txt
        outputToFile(predictedLabels, "output2.txt");
    }

    /**
     * Writes the predictedLabels to a file.
     *
     * @param predictedLabels The array of predicted labels.
     * @param fileName        The name of the output file.
     */
    public static void outputToFile(int[] predictedLabels, String fileName) {
        try (PrintWriter writer = new PrintWriter(fileName)) {
            for (int i = 0; i < predictedLabels.length; i++) {
                writer.print(predictedLabels[i] + " ");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads data and labels from the given files.
     *
     * @param dataFile   The file containing data.
     * @param data       The 2D array to store data.
     * @param labelFile  The file containing labels.
     * @param labels     The array to store labels.
     */
    public static void loadDataLabels(String dataFile, double[][] data, String labelFile, int[] labels) {
        try (Scanner dataScan = new Scanner(new File(dataFile)); Scanner labelScan = new Scanner(new File(labelFile))) {
            for (int i = 0; i < data.length && dataScan.hasNext(); i++){
                for (int j = 0; j < data[0].length && dataScan.hasNextDouble(); j++){
                    data[i][j] = dataScan.nextDouble();
                }
            }
            for (int i = 0; i < labels.length && labelScan.hasNextInt(); i++){
                labels[i] = labelScan.nextInt();
            }
        } 
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * Normalizes the data using z-score normalization.
     *
     * @param data The 2D array representing the data.
     */
    public static void normalizeData(double[][] data) {
        // it will iterate through each column (feature)
        for (int j = 0; j < data[0].length; j++) {
            double sum = 0.0;
            double sumOfSquares = 0.0;
            // calculate the sum and sum of squares for each column
            for (int i = 0; i < data.length; i++) {
                sum += data[i][j];
                sumOfSquares += Math.pow(data[i][j], 2);
            }
            // it calculates the mean and standard deviation for each column
            double mean = sum / data.length;
            double standardDeviation = Math.sqrt(sumOfSquares / data.length - Math.pow(mean, 2));
            // normalizes the data
            // z = (x - mean) / standard deviation
            for (int i = 0; i < data.length; i++) {
                data[i][j] = (data[i][j] - mean) / standardDeviation;
            }
        }
    }

    /**
     * Calculates the Manhattan distance between two instances.
     *
     * @param case1 The first instance.
     * @param case2 The second instance.
     * @return The Manhattan distance between the instances.
     */
    public static double calculatingManhattanDistance(double[] case1, double[] case2) {
        double total = 0;
        for (int i = 0; i < case1.length; i++) {
            total = total + Math.abs(case1[i] - case2[i]);
        }
        return total;
    }

    /**
     * Finds the index of the largest distance in the array.
     *
     * @param distances The array of distances.
     * @return The index of the largest distance.
     */
    public static int findIndexOfLargestDistance(double[] distances) {
        int maxIndex = 0;
        for (int i = 1; i < distances.length; i++) {
            if (distances[i] > distances[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Finds the k nearest neighbors of a test instance using Manhattan distance.
     *
     * @param testInstance   The test instance.
     * @param train          The training data.
     * @param k              The number of neighbors to find.
     * @return The indices of the k nearest neighbors.
     */
    public static int[] checkNearestNeighbours(double[] testInstance, double[][] train, int k) {
        double[] distances = new double[train.length];
        int[] nearestNeighbors = new int[k];
        for (int i = 0; i < k; i++) {
            nearestNeighbors[i] = i;
            distances[i] = calculatingManhattanDistance(testInstance, train[i]);
        }
        for (int i = k; i < train.length; i++) {
            double currentDistance = calculatingManhattanDistance(testInstance, train[i]);
            int maxIndex = findIndexOfLargestDistance(distances);
            if (currentDistance < distances[maxIndex]) {
                nearestNeighbors[maxIndex] = i;
                distances[maxIndex] = currentDistance;
            }
            }
        return nearestNeighbors;
    }
    
    /**
     * Finds the predicted label of a test instance with the improved kNN approach.
     *
     * @param testInstance The test instance.
     * @param train        The training data.
     * @param trainLabels  The labels of the training data.
     * @param k            The number of neighbors to consider.
     * @return The predicted label.
     */
    public static int getPredictedLabelWithImprovedApproach(double[] testInstance, double[][] train, int[] trainLabels, int k) {
        int[] nearestNeighbors = checkNearestNeighbours(testInstance, train, k);
        Map<Integer, Integer> countLabels = new HashMap<>();
        for (int neighbor : nearestNeighbors) {
            int label = trainLabels[neighbor];
            countLabels.put(label, countLabels.getOrDefault(label, 0) + 1);
        }
        return Collections.max(countLabels.entrySet(), Map.Entry.comparingByValue()).getKey();
    }
}