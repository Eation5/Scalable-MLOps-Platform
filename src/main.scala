// Scalable Mlops Platform

/*
A comprehensive MLOps platform built with Scala and Apache Spark for scalable machine learning model deployment, monitoring, and lifecycle management.
*/

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object Scalablemlopsplatform {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Scalable Mlops Platform App")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    println("Loading simulated data...")
    val data = Seq(
      (1.0, 2.0, 0.0),
      (2.0, 3.0, 0.0),
      (3.0, 4.0, 1.0),
      (4.0, 5.0, 1.0),
      (5.0, 6.0, 0.0),
      (6.0, 7.0, 1.0),
      (7.0, 8.0, 0.0),
      (8.0, 9.0, 1.0),
      (9.0, 10.0, 0.0),
      (10.0, 11.0, 1.0)
    ).toDF("feature1", "feature2", "label")

    println("Data loaded successfully.")

    println("Preprocessing data...")
    val assembler = new VectorAssembler()
      .setInputCols(Array("feature1", "feature2"))
      .setOutputCol("features")

    val featureData = assembler.transform(data)
    println("Data preprocessing complete.")

    val Array(trainingData, testData) = featureData.randomSplit(Array(0.8, 0.2), seed = 1234L)

    println("Training model...")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("binomial")

    val lrModel = lr.fit(trainingData)
    println("Model training complete.")

    println("Evaluating model...")
    val predictions = lrModel.transform(testData)
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Model accuracy (Area Under ROC): $accuracy")
    println("ML pipeline execution finished.")

    spark.stop()
  }
}
