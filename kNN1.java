import java.io.*;
import java.util.*;

/*
 * 
 * This class implements a k-Nearest Neighbors (kNN) algorithm.
 * The methods in this class include:
 * 1. loadDataLabels: loads data and labels from files
 * 2. calculatingEuclideanDistance: calculates the euclidean distance
 * 3. findIndexOfLargestDistance: finds the index of the largest distance in the array
 * 4. checkNearestNeighbours: finds the k nearest neighbors of a test instance
 * 5. getPredictedLabel: finds the predicted label of a test instance
 * 6. outputToFile: writes the predicted labels to a file
 * 7. main: main method
 * 
 * @author SS2697
 * @version 1.0
 * @date 2023/12/06
 * 
 */

public class kNN1 {

    /**
     * The main method for executing kNN prediction.
     * Loads data, performs kNN prediction with k = 1, calculates accuracy, and outputs results.
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
        int k = 1;
        int correct = 0;

        loadDataLabels("train_data.txt", train, "train_label.txt", trainLabel);
        loadDataLabels("test_data.txt", test, "test_label.txt", testLabels);


        // Perform kNN prediction with k = 1
        // will take single nearest neighbor
        for (int i = 0; i < TEST_SIZE; i++) {
            predictedLabels[i] = getPredictedLabel(test[i], train, trainLabel, k);
        }

        // Calculate accuracy and display it
        for (int i = 0; i < TEST_SIZE; i++) {
            if (predictedLabels[i] == testLabels[i]) {
                correct++;
            }
        }
        double accuracy = (double) correct / TEST_SIZE * 100;
        System.out.println("Accuracy: " + accuracy + "%");

        // output the predicted labels to file called output1.txt
        outputToFile(predictedLabels, "output1.txt");
        
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
     * Calculates the Euclidean distance between two instances.
     *
     * @param case1 The first instance.
     * @param case2 The second instance.
     * @return The Euclidean distance between the instances.
     */
    public static double euclideanDistanceCalculation(double[] case1, double[] case2) {
        double total = 0;
        for (int i = 0; i < case1.length; i++) {
            total = total + Math.pow(case1[i] - case2[i], 2);
        }
        return Math.sqrt(total);
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
     * Finds the k nearest neighbors of a test instance.
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
            distances[i] = euclideanDistanceCalculation(testInstance, train[i]);
        }
        for (int i = k; i < train.length; i++) {
            double currentDistance = euclideanDistanceCalculation(testInstance, train[i]);
            int maxIndex = findIndexOfLargestDistance(distances);
            if (currentDistance < distances[maxIndex]) {
                nearestNeighbors[maxIndex] = i;
                distances[maxIndex] = currentDistance;
            }
        }
        return nearestNeighbors;
    }          

    /**
     * Finds the predicted label of a test instance using kNN.
     *
     * @param testInstance The test instance.
     * @param train        The training data.
     * @param trainLabels  The labels of the training data.
     * @param k            The number of neighbors to consider.
     * @return The predicted label.
     */
    public static int getPredictedLabel(double[] testInstance, double[][] train, int[] trainLabels, int k) {
        Map<Integer, Integer> countLabels = new HashMap<>();
        int mostFrequentLabel = -1;
        int maxCount = -1;
        int[] nearestNeighbors = checkNearestNeighbours(testInstance, train, k);
        for (int neighbor : nearestNeighbors) {
            int label = trainLabels[neighbor];
            countLabels.put(label, countLabels.getOrDefault(label, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> currentEntry : countLabels.entrySet()) {
            Integer currentCount = currentEntry.getValue();
            if (currentCount > maxCount) {
                mostFrequentLabel = currentEntry.getKey();
                maxCount = currentCount;
                }
        }
        return mostFrequentLabel;
    }
}