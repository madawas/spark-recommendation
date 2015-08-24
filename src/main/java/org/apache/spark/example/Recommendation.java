package org.apache.spark.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Recommendation {
    public static final String LINE_SEPERATOR = "::";
    public static final String RESOURCE_PATH = "/home/madawa/WSO2/usb/data/movielens/medium/";
    public static final String RATINGS_FILE_NAME = "ratings.dat";
    public static final String MOVIES_FILE_NAME = "movies.dat";
    public static final String APP_NAME = "MovieRecommendation";
    public static final String CLUSTER = "local";
    private static JavaSparkContext sc;

    public static void main(String[] args) {
        //Setting up log levels
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        //Initializing Spark
        SparkConf conf = new SparkConf().setAppName(APP_NAME).setMaster(CLUSTER);
        sc = new JavaSparkContext(conf);

        //Reading external data
        final JavaRDD<String> ratingData = sc.textFile(RESOURCE_PATH + RATINGS_FILE_NAME);
        JavaRDD<String> productData = sc.textFile(RESOURCE_PATH + MOVIES_FILE_NAME);

        JavaRDD<Tuple2<Integer, Rating>> ratings = ratingData.map(
                new Function<String, Tuple2<Integer, Rating>>() {
                    public Tuple2<Integer, Rating> call(String s) throws Exception {
                        String[] row = s.split(LINE_SEPERATOR);
                        Integer cacheStamp = Integer.parseInt(row[3]) % 10;
                        Rating rating = new Rating(Integer.parseInt(row[0]), Integer.parseInt(row[1]), Double.parseDouble(row[2]));
                        return new Tuple2<Integer, Rating>(cacheStamp, rating);
                    }
                }
        );

        Map<Integer, String> products = productData.mapToPair(
                new PairFunction<String, Integer, String>() {
                    public Tuple2<Integer, String> call(String s) throws Exception {
                        String[] sarray = s.split(LINE_SEPERATOR);
                        return new Tuple2<Integer, String>(Integer.parseInt(sarray[0]), sarray[1]);
                    }
                }
        ).collectAsMap();

        long ratingCount = ratings.count();
        long userCount = ratings.map(
                new Function<Tuple2<Integer, Rating>, Object>() {
                    public Object call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().user();
                    }
                }
        ).distinct().count();

        long movieCount = ratings.map(
                new Function<Tuple2<Integer, Rating>, Object>() {
                    public Object call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().product();
                    }
                }
        ).distinct().count();

        System.out.println("Got " + ratingCount + " ratings from "
                + userCount + " users on " + movieCount + " products.");

        //Splitting training data
        int numPartitions = 10;
        //training data set
        JavaRDD<Rating> training = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() < 6;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        ).repartition(numPartitions).cache();

        StorageLevel storageLevel = new StorageLevel();
        //validation data set
        JavaRDD<Rating> validation = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() >= 6 && tuple._1() < 8;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        ).repartition(numPartitions).persist(storageLevel);

        //test data set
        JavaRDD<Rating> test = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() >= 8;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        ).persist(storageLevel);

        long numTraining = training.count();
        long numValidation = validation.count();
        long numTest = test.count();

        System.out.println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest);

        //training model
        int[] ranks = {8, 12};
        float[] lambdas = {0.1f, 10.0f};
        int[] numIters = {10, 20};

        double bestValidationRmse = Double.MAX_VALUE;
        int bestRank = 0;
        float bestLambda = -1.0f;
        int bestNumIter = -1;
        MatrixFactorizationModel bestModel = null;

        for (int currentRank : ranks) {
            for (float currentLambda : lambdas) {
                for (int currentNumIter : numIters) {
                    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(training), currentRank, currentNumIter, currentLambda);

                    Double validationRmse = computeRMSE(model, validation);
                    System.out.println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
                            + currentRank + ", lambda = " + currentLambda + ", and numIter = " + currentNumIter + ".");

                    if (validationRmse < bestValidationRmse) {
                        bestModel = model;
                        bestValidationRmse = validationRmse;
                        bestRank = currentRank;
                        bestLambda = currentLambda;
                        bestNumIter = currentNumIter;
                    }
                }
            }
        }

        //Computing Root Mean Square Error in the test dataset
        Double testRmse = computeRMSE(bestModel, test);
        RDD<Tuple2<Object, double[]>> features = bestModel.productFeatures();
        System.out.println("Saving model");
        bestModel.save(sc.sc(), "/home/madawa/model");
        features.saveAsTextFile(RESOURCE_PATH + "features");
        System.out.println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda +
                           ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".");

        System.out.println("Loading model");
        bestModel = MatrixFactorizationModel.load(sc.sc(), "/home/madawa/model");
        List<Rating> recommendations = getRecommendations(1, bestModel, ratings, products);

        //Printing Recommendations
        for (Rating recommendation : recommendations) {
            if (products.containsKey(recommendation.product())) {
                System.out.println(recommendation.product() + " " + products.get(recommendation.product()));
            }
        }

    }

    /**
     * Calculating the Root Mean Squared Error
     *
     * @param model best model generated.
     * @param data  rating data.
     * @return      Root Mean Squared Error
     */
    public static Double computeRMSE(MatrixFactorizationModel model, JavaRDD<Rating> data) {
        JavaRDD<Tuple2<Object, Object>> userProducts = data.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );

        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                                return new Tuple2<Tuple2<Integer, Integer>, Double>(
                                        new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
                            }
                        }
                ));
        JavaRDD<Tuple2<Double, Double>> predictionsAndRatings =
                JavaPairRDD.fromJavaRDD(data.map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                                return new Tuple2<Tuple2<Integer, Integer>, Double>(
                                        new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
                            }
                        }
                )).join(predictions).values();

        double mse =  JavaDoubleRDD.fromRDD(predictionsAndRatings.map(
                new Function<Tuple2<Double, Double>, Object>() {
                    public Object call(Tuple2<Double, Double> pair) {
                        Double err = pair._1() - pair._2();
                        return err * err;
                    }
                }
        ).rdd()).mean();

        return Math.sqrt(mse);
    }

    /**
     * Returns the list of recommendations for a given user
     *
     * @param userId    user id.
     * @param model     best model.
     * @param ratings   rating data.
     * @param products  product list.
     * @return          The list of recommended products.
     */
    private static List<Rating> getRecommendations(final int userId, MatrixFactorizationModel model, JavaRDD<Tuple2<Integer, Rating>> ratings, Map<Integer, String> products) {
        List<Rating> recommendations;

        //Getting the users ratings
        JavaRDD<Rating> userRatings = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().user() == userId;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        );

        //Getting the product ID's of the products that user rated
        JavaRDD<Tuple2<Object, Object>> userProducts = userRatings.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );

        List<Integer> productSet = new ArrayList<Integer>();
        productSet.addAll(products.keySet());

        Iterator<Tuple2<Object, Object>> productIterator = userProducts.toLocalIterator();

        //Removing the user watched (rated) set from the all product set
        while(productIterator.hasNext()) {
            Integer movieId = (Integer)productIterator.next()._2();
            if(productSet.contains(movieId)){
                productSet.remove(movieId);
            }
        }

        JavaRDD<Integer> candidates = sc.parallelize(productSet);

        JavaRDD<Tuple2<Integer, Integer>> userCandidates = candidates.map(
                new Function<Integer, Tuple2<Integer, Integer>>() {
                    public Tuple2<Integer, Integer> call(Integer integer) throws Exception {
                        return new Tuple2<Integer, Integer>(userId, integer);
                    }
                }
        );

        //Predict recommendations for the given user
        recommendations = model.predict(JavaPairRDD.fromJavaRDD(userCandidates)).collect();

        //Sorting the recommended products and sort them according to the rating
        Collections.sort(recommendations, new Comparator<Rating>() {
            public int compare(Rating r1, Rating r2) {
                return r1.rating() < r2.rating() ? -1 : r1.rating() > r2.rating() ? 1 : 0;
            }
        });

        //get top 50 from the recommended products.
        recommendations = recommendations.subList(0, 50);

        return recommendations;
    }
}