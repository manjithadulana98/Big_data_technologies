import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{Tokenizer, HashingTF, IDF, StringIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._

val raw = spark.read
  .option("header", "true")
  .option("multiLine", true)
  .option("escape", "\"")
  .csv("C:/Msc/Big Data/mlib_assignment/Merged_dataset.csv")
  .cache()

val data = raw.select("artist_name", "track_name", "release_date", "genre", "lyrics")

val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 1234L)

val tokenizer = new Tokenizer().setInputCol("lyrics").setOutputCol("words")
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10000)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val labelIndexer = new StringIndexer().setInputCol("genre").setOutputCol("label")
val classifier = new NaiveBayes()

val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, labelIndexer, classifier))

val model = pipeline.fit(training)

val predictions = model.transform(test)

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")

val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
val f1 = evaluator.setMetricName("f1").evaluate(predictions)

println(f"Accuracy : ${accuracy * 100}%.2f%%")
println(f"Precision: ${precision * 100}%.2f%%")
println(f"Recall   : ${recall * 100}%.2f%%")
println(f"F1 Score : ${f1 * 100}%.2f%%")

val labelIndexerModel = model.stages(3).asInstanceOf[org.apache.spark.ml.feature.StringIndexerModel]
val genreLabels = labelIndexerModel.labels
genreLabels.zipWithIndex.foreach { case (g, i) => println(f"$i → $g") }

val confusion = predictions.groupBy("label", "prediction").count()
confusion.orderBy("label", "prediction").show(100, false)

model.write.overwrite().save("C:/Msc/Big Data/mlib_assignment/models/mendeley_model")
labelIndexerModel.write.overwrite().save("C:/Msc/Big Data/mlib_assignment/models/label_indexer")
