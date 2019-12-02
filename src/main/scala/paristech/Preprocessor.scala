package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrameNaFunctions
import org.apache.spark.sql.types._

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /*println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")*/
    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("data/train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    //Afficher un extrait du DataFrame
    df.show()

    //Afficher le schéma du DataFrame
    df.printSchema()

    //Assignation integer aux types qui semblent être des entiers
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    //Cleaning
    //Affichez une description statistique des colonnes de type Int
    dfCasted.select("goal", "backers_count", "final_status").describe().show

    // Visualisation des données des colonnes
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    //Suppression de la colonne disable_communication
    val df2: DataFrame = dfCasted.drop("disable_communication")

    //Suppression des données du futur on retire les colonnes backers_count et state_changed_at :
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    // Colonnes currency et country
    //lorsque country = "False" le country à l'air d'être dans currency
    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    //Création de deux udfs nommées udf_country et udf_currency telles que :
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    // Définitions des udfs
    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    // Affichez le nombre d’éléments de chaque classe (colonne final_status)
    dfCountry.groupBy("final_status").count.orderBy($"count".desc).show(100)

    //Lignes final_status vaut 0 (Fail) ou 1 (Success)
    val dfCountryStatus: DataFrame = dfCountry.filter($"final_status" === 0 || $"final_status" === 1 )

    //Ajouter et manipuler des colonnes
    // Ajout de days_campaign
    // Ajout colonne hours_prepa
    val dfCountryStatusNew: DataFrame = dfCountryStatus
      .withColumn("days_campaign", datediff(from_unixtime($"deadline"),from_unixtime($"launched_at")))
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600, 3))
      .drop("deadline", "launched_at", "created_at")

    // Transformer contenu colonnes name, desc, et keywords en miniscule
    val dfCountryMin: DataFrame = dfCountryStatusNew
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))
      .withColumn("text", concat($"name",lit(' '), $"desc", lit(' '),$"keywords" ))

    dfCountryMin.show(10)

    //Remplacer les valeurs nulles par des valeurs spécifiques
    val dfCountryNa = dfCountryMin
      .na.fill(-1,Seq("goal"))
      .na.fill(-1,Seq("days_campaign"))
      .na.fill(-1,Seq("hours_prepa"))
      .na.fill("unknown",Seq("country2"))
      .na.fill("unknown",Seq("currency2"))
      .na.fill("unknown",Seq("text"))

    //Sauvegarde du DataFrame en parquet
    dfCountryNa.write.parquet("data/train_clean_PF.parquet")

  }
}
