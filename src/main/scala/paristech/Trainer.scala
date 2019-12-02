package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, DataFrameNaFunctions, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF,  RegexTokenizer, StopWordsRemover, Tokenizer, StringIndexer, OneHotEncoderEstimator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{LogisticRegression, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    //println("hello world ! from Trainer")
    val dfClean: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .parquet("data/train_clean_PF.parquet")

    //Stage1: séparation des textes en mots (ou tokens) avec un tokenizer
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //Stage 2 : retirer les stop words
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // Application algorithme TF-IDF
    val tokenized: DataFrame = tokenizer.transform(dfClean) //Transformation des phrases en liste de mots
    val wordsData: DataFrame = remover.transform(tokenized) // Suppression des StopWords

    val cvModel = new CountVectorizer() // Extracts a vocabulary from document collections and generates a CountVectorizerModel.
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")

    val featurizedData: DataFrame = cvModel.fit(wordsData).transform(wordsData) //application du cvModel

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("tfidf") //IDF is an Estimator which is fit on a dataset and produces an IDFModel.

    val idfModel = idf.fit(featurizedData)

    val rescaledData: DataFrame = idfModel.transform(featurizedData)

    //rescaledData.show()

    // Conversion des variables catégorielles en variables numériques   (utiliser StringIndexer de MLib)
    //Stage 5 : convertir country2 et currency2 en quantités numériques

    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")


    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")


    val indexedCountry: DataFrame = indexerCountry.fit(rescaledData).transform(rescaledData)

    val indexedData: DataFrame = indexerCurrency.fit(indexedCountry).transform(indexedCountry)

    //indexedData.show()

    //Stages 7 et 8: One-Hot encoder ces deux catégories

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))


    val encoded = encoder.fit(indexedData).transform(indexedData)
    //encoded.show()

    //Mettre les données sous une forme utilisable par Spark.ML
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    val output = assembler.transform(encoded)

    //Stage 10 : créer/instancier le modèle de classification
     val lr = new LogisticRegression()
       //.setElasticNetParam(0.0)
       .setFitIntercept(true)
       .setFeaturesCol("features")
       .setLabelCol("final_status")
       .setStandardization(true)
       .setPredictionCol("predictions")
       .setRawPredictionCol("raw_predictions")
       .setThresholds(Array(0.7, 0.3))
       .setTol(1.0e-6)
       .setMaxIter(20)

    //Création du Pipeline
     val pipeline = new Pipeline()
       .setStages(Array(tokenizer, remover, cvModel, idf, indexerCountry, indexerCurrency, encoder, assembler, lr))

    //Entraînement, test, et sauvegarde du modèle
     val Array(training, test) = dfClean.randomSplit(Array(0.9, 0.1))

    // Entraînement du modèle de regression logistique
     val model = pipeline.fit(training)

    // Test du modèle
    val dfWithSimplePredictions = model.transform(test)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(dfWithSimplePredictions)
    println("Test Error = " + (1.0 - accuracy))

    // Réglage des hyper-paramètres (a.k.a. tuning) du modèle
    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, (55.0 to 95.0 by 20.0).toArray)
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4,10e-2))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.7)

    val  model_tvs = trainValidationSplit.fit(training)
    val dfWithPredictions = model_tvs.transform(test)
    dfWithPredictions.groupBy("final_status", "predictions").count.show()
    val accuracy_tvs = evaluator.evaluate(dfWithPredictions)
    println("Test Error = " + (1.0 - accuracy_tvs))

    //Sauvegarde du modèle de régression logistique
    model_tvs.write.overwrite.save("models/model_lr_tvs")



    //Supplément (Test du classifieur DecisionTreeClassifier pour améliorer la prédiction)
    val dt = new  DecisionTreeClassifier()
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")

    //Définition d'un pipeline avec le DecisionTreeClassifier
    val pipeline_dt = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, indexerCountry, indexerCurrency, encoder, assembler, dt))

    //Fit sur les données de training
    val bc_model = pipeline_dt.fit(training)

    // Tranform des données test
    val dfWithSimplePredictionsDt = bc_model.transform(test)
    dfWithSimplePredictionsDt.groupBy("final_status", "predictions").count.show()

    // Select (prediction, true label) and compute test error.
    val evaluator_dt = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("accuracy")

    val accuracy_dt = evaluator_dt.evaluate(dfWithSimplePredictionsDt)
    println("Test Error = " + (1.0 - accuracy_dt))

    //Sauvegarde du modèle DecisionTreeClassifier
    bc_model.write.overwrite.save("models/decision_tree_model") //Ce modèle améliore légèrement notre précision


    // Amélioration du modèle DecisionTree
    //Test d'une grille de paramètres
    val dtparamGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth,(1 to 11 by 1).toArray)
      .addGrid(dt.maxBins, Array(2,10,20,40,60,80,100))
      .build()

    val dt_tvs = new TrainValidationSplit()
      .setEstimator(pipeline_dt)
      .setEvaluator(evaluator_dt)
      .setEstimatorParamMaps(dtparamGrid)
      .setTrainRatio(0.7)

    val  model_dt_tvs = dt_tvs.fit(training)

    val dfWithPredictionsDt = model_dt_tvs.transform(test)

    dfWithPredictionsDt.groupBy("final_status", "predictions").count.show()

    val accuracy_dt_tvs = evaluator_dt.evaluate(dfWithPredictionsDt)
    println("Test Error = " + (1.0 - accuracy_dt_tvs))

    //Sauvegarde du meilleur modèle de TrainValidation pour l'algo DecisionTree
    model_dt_tvs.write.overwrite.save("models/decision_tree_tvs_model")

  }
}
